import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid
import torch.nn.functional as F

import json
import os
from typing import List, Union
import random
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, "/home/et21-lijl/Documents/recognize-anything/")
from ram.models import ram, ram_plus
from ram.utils import openset_utils

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def forward_ram_plus(model: Module, imgs: Tensor) -> Tensor:
    image_embeds = model.image_proj(model.visual_encoder(imgs.to(device)))
    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long).to(device)

    image_cls_embeds = image_embeds[:, 0, :]
    image_spatial_embeds = image_embeds[:, 1:, :]

    bs = image_spatial_embeds.shape[0]

    des_per_class = int(model.label_embed.shape[0] / model.num_class)

    image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
    reweight_scale = model.reweight_scale.exp()
    logits_per_image = (reweight_scale * image_cls_embeds @ model.label_embed.t())
    logits_per_image = logits_per_image.view(bs, -1,des_per_class)

    weight_normalized = F.softmax(logits_per_image, dim=2)
    label_embed_reweight = torch.empty(bs, model.num_class, 512).cuda()
    weight_normalized = F.softmax(logits_per_image, dim=2)
    label_embed_reweight = torch.empty(bs, model.num_class, 512).cuda()
    for i in range(bs):
        reshaped_value = model.label_embed.view(-1, des_per_class, 512)
        product = weight_normalized[i].unsqueeze(-1) * reshaped_value
        label_embed_reweight[i] = product.sum(dim=1)

    label_embed = relu(model.wordvec_proj(label_embed_reweight))

    tagging_embed = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    ) 
    # tagging_embed[0] (bs, len(taglist), 768)
    # tagging_embed[1] None
    # tagging_embed[2] ((bs, 4, len(taglist), 145), (bs, 4, len(taglist), 145)) all of the attention weights
    # tagging_embed[3] None
    return sigmoid(model.fc(tagging_embed[0]).squeeze(-1)), tagging_embed[0], tagging_embed[2], image_embeds


@torch.no_grad()
def forward_ram(model: Module, imgs: Tensor, device: str) -> Tensor:
    image_embeds = model.image_proj(model.visual_encoder(imgs.to(device)))
    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long).to(device)
    label_embed = relu(model.wordvec_proj(model.label_embed)).unsqueeze(0)\
        .repeat(imgs.shape[0], 1, 1)
    tagging_embed = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )
    return sigmoid(model.fc(tagging_embed[0]).squeeze(-1)), tagging_embed[0], tagging_embed[2], image_embeds
    # return sigmoid(model.fc(tagging_embed[0]).squeeze(-1)), tagging_embed[0], label_embed


@torch.no_grad()
def forward_tag2text(
    model: Module,
    class_idxs: List[int],
    imgs: Tensor,
    device: str
) -> Tensor:
    image_embeds = model.visual_encoder(imgs.to(device))
    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long).to(device)
    label_embed = model.label_embed.weight.unsqueeze(0)\
        .repeat(imgs.shape[0], 1, 1)
    tagging_embed, _ = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )
    return sigmoid(model.fc(tagging_embed))[:, class_idxs]


from clip import clip
def build_openset_label_embedding(llm_tag_des):
    print("Creating pretrained CLIP model")
    model, _ = clip.load("ViT-B/16")
    categories = []

    run_on_gpu = torch.cuda.is_available()

    # llm_tag_des = [x.split(",")[1].strip() for x in llm_tag_des]
    with torch.no_grad():
        openset_label_embedding = []
        for des in tqdm(llm_tag_des):

            texts = clip.tokenize(des, truncate=True)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
                model = model.cuda()
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)
        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)
        if run_on_gpu:
            openset_label_embedding = openset_label_embedding.cuda()

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding, categories


def load_ram_plus(
    backbone: str,
    checkpoint: str,
    input_size: int,
    taglist: List[str],
    open_set: bool,
    class_idxs: List[int],
) -> Module:
    model = ram_plus(pretrained=checkpoint, image_size=input_size, vit=backbone)
    # trim taglist for faster inference
    if open_set:
        print("Building tag embeddings ...")
        # label_embed, _ = build_openset_llm_label_embedding(tag_des)
        label_embed, _ = build_openset_label_embedding(taglist)
        model.label_embed = Parameter(label_embed.float())
        model.num_class = len(taglist)
    else:
        model.label_embed = Parameter(model.label_embed.data.reshape(model.num_class,51,512)[class_idxs, :, :].reshape(len(class_idxs)*51, 512))
        model.num_class = len(class_idxs)
    return model.to(device).eval()


def load_ram(
    backbone: str,
    checkpoint: str,
    input_size: int,
    taglist: List[str],
    open_set: bool,
    class_idxs: List[int],
    device:str
) -> Module:
    model = ram(pretrained=checkpoint, image_size=input_size, vit=backbone)
    # trim taglist for faster inference
    if open_set:
        print("Building tag embeddings ...")
        label_embed, _ = openset_utils.build_openset_label_embedding(taglist)
        model.label_embed = Parameter(label_embed.float())
    else:
        model.label_embed = Parameter(model.label_embed[class_idxs, :])

    return model.to(device).eval()


def get_pos_and_neg(taglist, ind_taglist, thresholds, logits, tagging_embed, tars, tau):
    bs,num_tag,dimen = tagging_embed.shape
    pos_index        = [taglist.index(ind_taglist[tar]) for tar in tars]
    pos_element      = tagging_embed[range(bs), pos_index]

    logits[range(bs), pos_index] = 0
    logits = logits.flatten()

    tagging_embed = tagging_embed.reshape(-1, dimen)
    neg_group     = tagging_embed[logits > thresholds]
    ab_mask       = torch.cosine_similarity(
        pos_element.unsqueeze(1), neg_group, dim=-1) > tau
    ab_mask       = ab_mask.sum(0) == 0
    neg_group     = neg_group[ab_mask]
    # print(neg_group.shape)
    
    return pos_element, neg_group


def get_pos_neg_group(tokens, tau):
    bs          = len(tokens)
    pos_element = [tokens[i][0:1] for i in range(bs)]
    pos_element = torch.vstack(pos_element)

    neg_group   = [tokens[i][1:] for i in range(bs)]
    neg_group   = torch.vstack(neg_group)
    
    ab_mask     = torch.cosine_similarity(
        pos_element.unsqueeze(1), neg_group, dim=-1) > tau
    ab_mask       = ab_mask.sum(0) == 0
    neg_group     = neg_group[ab_mask]
    # print(neg_group.shape)
    
    return pos_element, neg_group


def gen_samples(pos_element, neg_group, device, k=5):
    bs           = pos_element.shape[0]

    neg_samples  = [torch.vstack(random.sample(list(neg_group), k)).unsqueeze(0) for i in range(bs)]
    neg_samples  = torch.vstack(neg_samples)

    neg_elements = [torch.vstack(random.sample(list(neg_group), k-1)).unsqueeze(0) for i in range(bs)]
    neg_elements = torch.vstack(neg_elements)
    pos_samples  = torch.cat([pos_element.unsqueeze(1), neg_elements], dim=-2)
    rand_indices = torch.randperm(pos_samples.shape[1])
    pos_samples  = pos_samples[:, rand_indices]
    del neg_group

    samples = torch.cat([pos_samples, neg_samples], dim=0)
    labels  = torch.cat([torch.zeros(bs,device=device), torch.ones(bs,device=device)]).long()
    return samples, labels


def gen_samples_onestream(tokens, device, k, tau):
    bs          = len(tokens)
    pos_element = [tokens[i][0:1] for i in range(bs)]
    pos_element = torch.vstack(pos_element)

    neg_group   = [tokens[i][1:] for i in range(bs)]
    neg_group   = torch.vstack(neg_group)

    # neg_elements = [torch.vstack(random.sample(list(neg_group), k-1)).unsqueeze(0) for i in range(bs)]
    # neg_elements = torch.vstack(neg_elements)
    # pos_samples  = torch.cat([pos_element.unsqueeze(1), neg_elements], dim=-2)

    try:
        neg_elements = [torch.vstack(random.sample(list(tokens[i]), k-1)).unsqueeze(0) for i in range(bs)]
    except:
        neg_elements = [torch.vstack(random.sample(list(neg_group), k-1)).unsqueeze(0) for i in range(bs)]
    neg_elements = torch.vstack(neg_elements)
    pos_samples  = torch.cat([pos_element.unsqueeze(1), neg_elements], dim=-2)

    ab_mask     = torch.cosine_similarity(
        pos_element.unsqueeze(1), neg_group, dim=-1) > tau
    ab_mask       = ab_mask.sum(0) == 0
    neg_group     = neg_group[ab_mask]

    neg_samples  = [torch.vstack(random.sample(list(neg_group), k)).unsqueeze(0) for i in range(bs)]
    neg_samples  = torch.vstack(neg_samples)

    samples = torch.cat([pos_samples, neg_samples], dim=0)
    labels  = torch.cat([torch.zeros(bs,device=device), torch.ones(bs,device=device)]).long()
    return samples, labels


def get_prompts(classes, tars, prom_path):
    prompts = []
    for tar in tars:
        name = classes[tar]
        prompt = torch.load(os.path.join(prom_path, f"{name}_prompt.pth"))
        prompts.append(prompt)
    prompts = torch.vstack(prompts * 2).unsqueeze(1)
    prompts.requires_grad_()
    return prompts


class ConfigLoader:
    def __init__(
            self,
            architectures=None,
            attention_probs_dropout_prob=0.1,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=3072,
            layer_norm_eps=1e-12,
            max_position_embeddings=512,
            model_type="bert",
            num_attention_heads=12,
            num_hidden_layers=12,
            pad_token_id=0,
            type_vocab_size=2,
            vocab_size=30524,
            encoder_width=768,
            add_cross_attention=True,
            output_attentions=True, 
            is_cross_attention=True, **kwargs) -> None:

        self.architectures=architectures
        self.attention_probs_dropout_prob=attention_probs_dropout_prob
        self.hidden_act=hidden_act
        self.hidden_dropout_prob=hidden_dropout_prob
        self.hidden_size=hidden_size
        self.initializer_range=initializer_range
        self.intermediate_size=intermediate_size
        self.layer_norm_eps=layer_norm_eps
        self.max_position_embeddings=max_position_embeddings
        self.model_type=model_type
        self.num_attention_heads=num_attention_heads
        self.num_hidden_layers=num_hidden_layers
        self.pad_token_id=pad_token_id
        self.type_vocab_size=type_vocab_size
        self.vocab_size=vocab_size
        self.encoder_width=encoder_width
        self.add_cross_attention=add_cross_attention
        self.output_attentions=output_attentions
        self.is_cross_attention=is_cross_attention
        self.__dict__.update(kwargs)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)