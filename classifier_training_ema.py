import torch
import torch.nn.functional as F
import lightning as L
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import argparse
from tqdm import tqdm
from time import time
from argparse import ArgumentParser
from colorama import Fore, Style
from pathlib import Path

import utils.data_utils as data_utils
import utils.model_utils as model_utils
import utils.metric_utils as metric_utils
from models.classifier import Classifier


# device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = ArgumentParser()
    # data
    parser.add_argument("--img_root", type=str,
                        default="/path/to/imagenet/")
    parser.add_argument("--attn_root", type=str,
                        default="outputs/")
    parser.add_argument("--input-size",
                        type=int,
                        default=384)
    parser.add_argument("--shuffle", type=bool_flag, default="True")
    parser.add_argument("--num-class", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)

    # training setting
    parser.add_argument("--num_dev", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)

    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./outputs")

    args = parser.parse_args()
    return args


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def detector_training(args):
    sttime = time() #  init lightning 
    fabric = L.Fabric(accelerator="cuda", devices=args.num_dev, strategy="auto")
    fabric.print(Fore.RED + "==> Preparing lightning..", end="")
    fabric.launch()
    fabric.print(f"took {time()-sttime:.1f}s.")

    sttime = time() #  load Classifier 
    fabric.print("==> Preparing project model..", end="")
    config_path = "configs/classifier_config.json"
    config      = model_utils.ConfigLoader.from_json_file(config_path)
    proj        = Classifier(config)
    fabric.print(f"took {time()-sttime:.1f}s.")

    sttime = time() #  load training data
    fabric.print(Fore.YELLOW + "==> Preparing training data..", end="")
    loader = data_utils.get_imagenet_tokens_train_loader_classification(
        args.attn_root, 
        args.batch_size, args.num_workers, args.shuffle, config.num_attention_heads)
    loader = fabric.setup_dataloaders(loader)
    fabric.print(f"took {time()-sttime:.1f}s.")
    print("Data loaded num_samples:", len(loader.dataset))

    sttime = time() #  load optimizer and utils 
    fabric.print("==> Preparing optimizer and utils..", end="")
    emaloss   = EMALoss(args.num_class, config.hidden_size, config.num_pattern, alpha=0.999)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=args.epochs * len(loader), 
                                                           eta_min=args.lr_min)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = data_utils.get_logger(os.path.join(args.output_dir, "logger.log"))
    checkpoint_list = os.listdir(args.output_dir)
    checkpoint_list = [i for i in checkpoint_list if "checkpoint" in i]
    al_epoch = 0
    if len(checkpoint_list) > 1:
        al_epoch = [cp.split("_")[1].split(".")[0] \
                    for cp in checkpoint_list if "checkpoint" in cp]
        al_epoch.remove("init")
        al_epoch = [int(i) for i in al_epoch]
        al_epoch = int(max(al_epoch))

        cp_path  = os.path.join(args.output_dir, f"checkpoint_{al_epoch}.pth")
        state_dict = torch.load(cp_path, map_location="cuda")

        proj.load_state_dict(state_dict["projector"])
        emaloss.centers = state_dict["ema"]
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        fabric.print(f"Load from checkpoint {cp_path}")
    else:
        for (key, value) in vars(args).items():
            logger.info(f"{key}: {value}")
    proj, emaloss, optimizer = fabric.setup(proj, emaloss, optimizer)
    emaloss = fabric.setup(emaloss)

    seed = 42
    fabric.seed_everything(seed)
    fabric.save(os.path.join(args.output_dir, "checkpoint_init.pth"),
                {"projector":proj.state_dict()})
    losses, cls_acc = metric_utils.AverageMeter(), metric_utils.Accuracy()
    fabric.print(f"took {time()-sttime:.1f}s.\n")

    # start training
    fabric.print(Fore.GREEN + "==> Training.." + Style.RESET_ALL)
    if torch.cuda.current_device() == 0:
        logger.info("\nStart training!")
    proj.train()

    for epoch in range(al_epoch+1, args.epochs):
        losses.reset(); cls_acc.reset()

        with tqdm(loader, desc=f"[{epoch:03d}]/[{args.epochs}] training", 
                  postfix={"loss":0.000, "acc":0.000}, 
                  disable=(torch.cuda.current_device() != 0)) as t:

            for data in t:
                image_embed_batch, attn_mask, cls_labels = data
                optimizer.zero_grad()
                feats  = proj.get_penultimate(image_embed_batch, mask=attn_mask)
                loss   = emaloss(feats, cls_labels.flatten())
                logits = proj(image_embed_batch, mask=attn_mask)
                loss_c = F.cross_entropy(logits, cls_labels.flatten())

                loss = torch.clamp(loss*0.1+loss_c, min=1e-5, max=1e+3)
                losses.update(loss.item())

                cls_acc.update(logits, cls_labels)

                fabric.backward(loss)
                optimizer.step()
                scheduler.step()

                t.set_postfix(
                            loss=round(losses.avg, 3), 
                            cls_acc=round(cls_acc.get_top1(), 3),
                            lr=round(optimizer.param_groups[0]["lr"], 6))

        fabric.save(os.path.join(args.output_dir, f"checkpoint_{epoch}.pth"),
                    {"projector":proj.state_dict(),
                     "ema":emaloss.centers,
                    "optimizer":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict()})

        if torch.cuda.current_device() == 0:
            logger.info('Epoch:[{}/{}]\t pos_loss={:.3f} acc={:.3f} lr={:.6f}'.format(
                epoch+1, 
                args.epochs, 
                losses.avg,
                cls_acc.get_top1(),
                optimizer.param_groups[0]["lr"]))

    if torch.cuda.current_device() == 0:
        logger.info("Finish training!")


class EMALoss(torch.nn.Module):
    def __init__(self, num_class, dims, num_heads, alpha=0.999):
        super().__init__()
        self.alpha = alpha
        self.num_class = num_class
        self.dims = dims
        self.num_heads = num_heads
        self.register_buffer("centers", torch.randn(num_class, num_heads, dims//num_heads))

    def forward(self, x, labels):
        centers = self.centers[labels]
        x = x.reshape(-1, self.num_heads, self.dims//self.num_heads)
        loss = torch.nn.functional.mse_loss(x, centers, reduction="none").mean(-1)
        loss = loss.sum(-1).mean()
        with torch.no_grad():
            self.centers[labels] = self.alpha * centers + (1 - self.alpha) * x
        return loss

    def weight_forward(self, x, weights, labels):
        centers = self.centers[labels]
        x = x.reshape(-1, self.num_heads, self.dims//self.num_heads)
        loss = torch.nn.functional.mse_loss(x, centers, reduction="none").mean(-1)
        weights = torch.softmax(weights, dim=-1)
        loss = (loss * weights).sum()
        with torch.no_grad():
            self.centers[labels] = self.alpha * centers + (1 - self.alpha) * x
        return loss


if __name__ == "__main__":
    args = parse_args()
    detector_training(args)
