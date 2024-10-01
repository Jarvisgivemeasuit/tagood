import torch
import lightning as L

import os
import argparse
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from colorama import Fore, Style

import utils.data_utils as data_utils
import utils.model_utils as model_utils
from ram.models import ram

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = ArgumentParser()
    # tagging model
    parser.add_argument("--tagging-type",
                        type=str,
                        choices=("ram", "ram_plus", "tag2text"),
                        required=True)
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True)
    parser.add_argument("--backbone",
                        type=str,
                        choices=("swin_l", "swin_b"),
                        default=None,
                        help="If `None`, will judge from `--model-type`")
    parser.add_argument("--open-set",
                        action="store_true",
                        help=(
                            "Treat all categories in the taglist file as "
                            "unseen and perform open-set classification. Only "
                            "works with RAM."
                        ))
    # threshold
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold",
                       type=float,
                       default=None,
                       help=(
                           "Use custom threshold for all classes. Mutually "
                           "exclusive with `--threshold-file`. If both "
                           "`--threshold` and `--threshold-file` is `None`, "
                           "will use a default threshold setting."
                       ))
    group.add_argument("--threshold-file",
                       type=str,
                       default=None,
                       help=(
                           "Use custom class-wise thresholds by providing a "
                           "text file. Each line is a float-type threshold, "
                           "following the order of the tags in taglist file. "
                           "See `ram/data/ram_tag_list_threshold.txt` as an "
                           "example. Mutually exclusive with `--threshold`. "
                           "If both `--threshold` and `--threshold-file` is "
                           "`None`, will use default threshold setting."
                       ))

    # data
    parser.add_argument("--dataset",
                        type=str,
                        choices=("ImageNet", "iNaturalist",
                                 "SUN", "Places","Textures", 
                                 "ImageNet-O", "OpenImage"),
                        required=True)
    parser.add_argument("--mode",
                        type=str, choices=["train", "val"],
                        required=True)
    parser.add_argument("--img_root", type=str,
                        default="/path/to/dataset/")
    parser.add_argument("--input-size",
                        type=int,
                        default=384)
    parser.add_argument("--shuffle", type=bool_flag, default="True")
    parser.add_argument("--num-class", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)

    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--num_dev", type=int, default=1)


    args = parser.parse_args()

    # post process and validity check
    args.tagging_type = args.tagging_type.lower()

    assert not (args.tagging_type == "tag2text" and args.open_set)

    if args.backbone is None:
        args.backbone = "swin_l" if args.tagging_type == "ram" or args.tagging_type == "ram_plus" else "swin_b"

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


def get_batch_info(ind2tag_idxs, thresholds, logits, tagging_embed, device):
    bs = logits.shape[0]

    thresholds_ = torch.tensor(thresholds, device=device).unsqueeze(0).expand(bs, len(thresholds))
    raw_indices, col_indices = torch.where(logits > thresholds_)

    batch_tokens = []
    batch_tokens_idx = []
    batch_tokens_logits = []

    batch_can_logit, batch_can_idx = torch.max(logits[:, ind2tag_idxs], dim=-1)
    batch_can_idx = [ind2tag_idxs[can_idx] for can_idx in batch_can_idx]
    batch_candidate = tagging_embed[range(bs), batch_can_idx]

    for i in range(bs):
        tokens_idx = col_indices[raw_indices==i]

        if len(tokens_idx) == 0:
            tokens_logits   = batch_can_logit[i]
            tokens_idx      = torch.tensor([batch_can_idx[i]])
            tokens          = tagging_embed[i, tokens_idx]
        else:
            tokens        = tagging_embed[i, tokens_idx]
            tokens_logits = logits[i, tokens_idx]

        batch_tokens.append(tokens)
        batch_tokens_idx.append(tokens_idx)
        batch_tokens_logits.append(tokens_logits)

    return (batch_tokens, batch_tokens_idx, batch_tokens_logits,
            batch_candidate, batch_can_idx, batch_can_logit)


def gen_data_tokens(args):
    fabric = L.Fabric(accelerator="cuda", devices=args.num_dev, strategy="auto")
    fabric.print(Fore.BLUE + "==> Preparing lightning..\n")
    fabric.launch()

    fabric.print(Fore.YELLOW + "==> Preparing tag lists..")
    tag_file = "texts/ram_tag_list.txt"
    with open(tag_file, "r", encoding="utf-8") as f:
        taglist = [line.strip() for line in f]

    ind_tag_file = "texts/ind_tag_list.txt"
    with open(ind_tag_file, "r", encoding="utf-8") as f:
        ind_taglist = [line.strip().split(",") for line in f]
        
    class_names = []
    class_tags  = []
    for info_list in ind_taglist:
        class_name = info_list[0].split(" ")[1]
        class_tag  = info_list[1]
        class_names.append(class_name)
        class_tags.append(class_tag)
    ind2tag_idxs = [taglist.index(tag) for tag in class_tags]

    class_idxs = data_utils.get_class_idxs(
        tagging_type=args.tagging_type,
        open_set=args.open_set,
        taglist=taglist
    )

    thresholds = data_utils.load_thresholds(
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        tagging_type=args.tagging_type,
        open_set=args.open_set,
        class_idxs=class_idxs,
        num_classes=len(taglist)
    )

    fabric.print("==> Preparing tagging model.." + Style.RESET_ALL)
    # load model
    if args.tagging_type == "ram":
        fabric.print('load checkpoint from %s\n' % args.checkpoint)

        model = model_utils.load_ram(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size,
            taglist=taglist,
            open_set=args.open_set,
            class_idxs=class_idxs,
            device=device
        )
    elif args.tagging_type == "ram_plus":
        fabric.print('load checkpoint from %s\n' % args.checkpoint)
        model = model_utils.load_ram_plus(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size,
            taglist=taglist,
            open_set=args.open_set,
            class_idxs=class_idxs,
            # device=device
        )
    else:
        model = ram.load_tag2text(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size
        )
    model = fabric.setup(model)
    model.eval()
    thres = 0.5

    fabric.print(Fore.GREEN + "==> Generating.." + Style.RESET_ALL)
    if args.dataset == "ImageNet":
        for cls_idx in range(args.num_class):
            if args.mode == "train":
                loader = data_utils.get_imagenet_one_class_train_loader(args.img_root, 
                                                                        cls_idx, 
                                                                        args.batch_size, 
                                                                        args.num_workers, 
                                                                        args.input_size, 
                                                                        args.shuffle,
                                                                        get_name=True)
            elif args.mode == "val":
                loader = data_utils.get_imagenet_one_class_val_loader(args.img_root, 
                                                                        cls_idx, 
                                                                        args.batch_size, 
                                                                        args.num_workers, 
                                                                        args.input_size, 
                                                                        args.shuffle,
                                                                        get_name=True)

            loader     = fabric.setup_dataloaders(loader)
            class_name = loader.dataset.class_name
            path       = os.path.join(args.output_dir, class_name)
            Path(path).mkdir(parents=True, exist_ok=True)

            for batch in tqdm(loader, 
                            desc=f"[{cls_idx + 1:04d}]/[{args.num_class}] generate tokens",
                            disable=(torch.cuda.current_device() != 0)):
                imgs, file_names = batch
                del batch
                with torch.no_grad():
                    logits, tagging_embed, attn_map, image_embed = model_utils.forward_ram_plus(model, imgs)
                    out = get_batch_info(ind2tag_idxs, thresholds, logits, tagging_embed, fabric.device)

                (batch_tokens, batch_tokens_idx, batch_tokens_logits,
                batch_candidate, batch_can_idx, batch_can_logit) = out
                del out

                bs = len(file_names)
                for i in range(bs):
                    tokens_idx = batch_tokens_idx[i]
                    can_idx = batch_can_idx[i]
                    can_logit = batch_can_logit[i]
                    save_path = os.path.join(path, f"{file_names[i]}.pt")
                    res_idx = list(set(tokens_idx.tolist()) & set(ind2tag_idxs))

                    if res_idx == []:
                        max_logit_idx = can_idx
                        max_logit = can_logit
                        attn = attn_map[0][i, :, max_logit_idx].sum(0)
                        attn = (attn - attn.min()) / (attn.max() - attn.min())
                        assert max_logit_idx in ind2tag_idxs
                        im_embed = image_embed[i][attn > thres]
                        torch.save([max_logit_idx,
                                max_logit,
                                attn, im_embed], save_path)
                    else:
                        res_logits = logits[i][res_idx]
                        attn = attn_map[0][i, :, res_idx].sum(0)
                        attn = (attn - attn.min(1)[0].unsqueeze(1)) / (
                            attn.max(1)[0].unsqueeze(1) - attn.min(1)[0].unsqueeze(1))
                        attn = attn.max(0)[0]
                        # attn = torch.any(attn > thres, dim=-2)
                        im_embed = image_embed[i][attn > thres]
                        torch.save([res_idx,
                                    res_logits,
                                    attn, 
                                    im_embed], save_path)

    else:
        loader = data_utils.get_ood_loaders(args.img_root, 
                                            args.batch_size, 
                                            args.num_workers, 
                                            args.input_size)[args.dataset]
        loader = fabric.setup_dataloaders(loader)
        path   = os.path.join(args.output_dir, args.dataset)
        Path(path).mkdir(parents=True, exist_ok=True)

        batch_num = 0
        for batch in tqdm(loader, 
                            desc=f"[{args.dataset}] generate tokens",
                            disable=(torch.cuda.current_device() != 0)):
            imgs, file_names = batch
            del batch
            with torch.no_grad():
                logits, tagging_embed, attn_map, image_embed = model_utils.forward_ram(model, imgs, device)
                out = get_batch_info(ind2tag_idxs, thresholds, logits, tagging_embed, fabric.device)

            (batch_tokens, batch_tokens_idx, batch_tokens_logits,
            batch_candidate, batch_can_idx, batch_can_logit) = out
            del out

            bs = imgs.shape[0]
            for i in range(bs):
                tokens = batch_tokens[i]
                tokens_idx = batch_tokens_idx[i]
                token_logits = batch_tokens_logits[i]
                candidate = batch_candidate[i]
                can_idx = batch_can_idx[i]
                can_logit = batch_can_logit[i]

                save_path = os.path.join(path, f"{file_names[i]}.pt")
                Path(os.path.join(*save_path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
                res_idx = list(set(tokens_idx.tolist()) & set(ind2tag_idxs))
                if res_idx == []:
                    max_logit_idx = []
                    max_logit = []
                    attn = []
                    im_embed = []
                    torch.save([max_logit_idx,
                            max_logit,
                            attn, im_embed], save_path)

                else:
                    res_logits = logits[i][res_idx]
                    attn = attn_map[0][i, :, res_idx].sum(0)
                    attn = (attn - attn.min(1)[0].unsqueeze(1)) / (
                        attn.max(1)[0].unsqueeze(1) - attn.min(1)[0].unsqueeze(1))
                    attn = attn.max(0)[0]

                    im_embed = image_embed[i][attn > thres]
                    torch.save([res_idx,
                                res_logits,
                                attn, 
                                im_embed], save_path)
            batch_num += 1
if __name__ == "__main__":
    args = parse_args()
    gen_data_tokens(args)
