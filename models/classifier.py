import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchsummary import summary
from torchvision.ops.misc import Conv2dNormActivation, MLP
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Dict
from collections import OrderedDict

import math
import sys
sys.path.append("..")
# from utils import model_utils


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False, attn_mask=mask)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length+1, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding[:, :input.size(1), :]
        x = self.dropout(input)
        for i in range(len(self.layers)):
            x = self.layers[i](x, mask)
        return self.ln(x)



class Classifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config     = config
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.image_proj = nn.Linear(config.encoder_width, config.hidden_size)
        self.layers     = Encoder(config.max_length,
                                     config.num_hidden_layers, 
                                     config.num_attention_heads,
                                     config.hidden_size, 
                                     config.hidden_size * 4, 
                                     config.hidden_dropout_prob, 
                                     config.attention_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.w_layer    = nn.Linear(config.hidden_size, config.num_pattern)

    def forward(self, attn_image_embeddings, mask=None):
        bs = attn_image_embeddings.shape[0]
        cls_token = self.cls_token.expand(bs, -1, -1)
        attn_image_embeddings = self.image_proj(attn_image_embeddings)
        attn_image_embeddings = torch.cat([cls_token, attn_image_embeddings], dim=1)
        attn_image_embeddings = self.layers(attn_image_embeddings, mask)

        output = attn_image_embeddings[:, 0, :]
        output = self.classifier(output)

        return output

    def forward_threshold(self, attn_image_embeddings, threshold, mask=None):
        bs = attn_image_embeddings.shape[0]
        cls_token = self.cls_token.expand(bs, -1, -1)
        attn_image_embeddings = self.image_proj(attn_image_embeddings)
        attn_image_embeddings = torch.cat([cls_token, attn_image_embeddings], dim=1)
        attn_image_embeddings = self.layers(attn_image_embeddings, mask)

        output = attn_image_embeddings[:, 0, :]
        output = output.clip(max=threshold)
        output = self.classifier(output)

        return output
    
    def get_penultimate(self, attn_image_embeddings, mask=None):
        bs = attn_image_embeddings.shape[0]
        cls_token = self.cls_token.expand(bs, -1, -1)
        attn_image_embeddings = self.image_proj(attn_image_embeddings)
        attn_image_embeddings = torch.cat([cls_token, attn_image_embeddings], dim=1)
        attn_image_embeddings = self.layers(attn_image_embeddings, mask)

        output = attn_image_embeddings[:, 0, :]
        return output
    

    def get_penultimate_weights(self, attn_image_embeddings=None):
        bs = attn_image_embeddings.shape[0]
        cls_token = self.cls_token.expand(bs, -1, -1)
        attn_image_embeddings = self.image_proj(attn_image_embeddings)
        attn_image_embeddings = torch.cat([cls_token, attn_image_embeddings], dim=1)
        attn_image_embeddings = self.layers(attn_image_embeddings)

        output = attn_image_embeddings[:, 0, :]
        weight = self.w_layer(output)
        return output, weight

# if __name__ == "__main__":
#     config_path = "/home/et21-lijl/Documents/multimodalood/configs/med_config.json"
#     config      = model_utils.ConfigLoader.from_json_file(config_path)
#     model       = DetectModel(config).cuda()
#     summary(model, [(5, 512), (1, 512)])