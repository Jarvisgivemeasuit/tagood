import torch
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
from typing import List, Optional

import utils.data_utils as data_utils
import utils.model_utils as model_utils
import utils.metric_utils as metric_utils
from models.classifier import Classifier

def parse_args():
    parser = ArgumentParser()
    # data
    parser.add_argument("--ind_root", type=str,
                        default="/path/to/ind_tokens/")
    parser.add_argument("--ood_root", type=str,
                        default="/path/to/ood_tokens/")

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
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs")
    parser.add_argument("--save-dir", type=str, default="./outputs")

    args = parser.parse_args()
    return args


def inference(args):
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
    ind_loader = data_utils.get_data_tokens_test_loader_classification(
        args.ind_root,
        args.batch_size, args.num_workers)
    ood_loaders = data_utils.get_ood_token_loaders_classification(
        args.ood_root, 
        args.batch_size, args.num_workers)
    ind_loader  = fabric.setup_dataloaders(ind_loader)
    ood_loaders = {k:fabric.setup_dataloaders(ood_loaders[k]) for k in ood_loaders}
    fabric.print(f"took {time()-sttime:.1f}s.")

    sttime = time() #  load state dict
    fabric.print("==> Preparing state dict..", end="")
    checkpoint_list = os.listdir(args.checkpoint_dir)
    checkpoint_list = [i for i in checkpoint_list if "checkpoint" in i]
    al_epoch = [cp.split("_")[1].split(".")[0] \
                for cp in checkpoint_list if "checkpoint" in cp]
    al_epoch.remove("init")
    al_epoch = [int(i) for i in al_epoch]
    al_epoch = int(max(al_epoch))

    cp_path  = os.path.join(args.checkpoint_dir, f"checkpoint_{al_epoch}.pth")
    state_dict = torch.load(cp_path, map_location="cuda")
    proj.load_state_dict(state_dict["projector"], strict=False)
    proj = fabric.setup(proj)
    proj.eval()
    centers = state_dict["ema"].cpu()
    fabric.print(f"took {time()-sttime:.1f}s.\n" + Style.RESET_ALL)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ind_scores = get_scores(ind_loader, proj, centers, args.save_dir, "ImageNet")
    for k in ood_loaders:
        ood_scores = get_scores(ood_loaders[k], proj, centers, args.save_dir, k)
        auroc, fpr95, aupr = metric_utils.get_results(ind_scores, ood_scores)
        fabric.print(f"{k} Results: AUROC:{auroc * 100:.2f}%, FPR95:{fpr95 * 100:.2f}%, AUPR:{aupr * 100:.2f}%")


def get_scores(loader, proj, centers, save_dir, dataset_name):
    scores, logits = [], []
    print(loader.dataset.__len__())
    with tqdm(loader, desc=f"calculating", 
                disable=(torch.cuda.current_device() != 0)) as t:
        for data in t:
            image_embed_batch, attn_mask_batch, names_batch, scores_batch, labels_batch = data
            bs = len(image_embed_batch)

            for i in range(bs):
                image_embed = image_embed_batch[i]
                name        = names_batch[i]
                score       = scores_batch[i]
                if len(image_embed) == 0:
                    scores.append(0)
                    continue
                with torch.no_grad():
                    feats = proj.get_penultimate(image_embed.unsqueeze(0), None)
                    feats = feats.cpu()
                    # feats, weight = proj.get_penultimate_weights(image_embed.unsqueeze(0))
                    # feats, weight = feats.cpu(), weight.cpu()
                    logit = proj(image_embed.unsqueeze(0), None).cpu()

                "euclidean distance"
                # dists = metric_utils.euclidean_distance(feats, centers)
                # scores.append(1/dists.min().item())

                "weighted euclidean distance"
                # feats = feats.reshape(-1, 4, 128)
                # weight = torch.softmax(weight, dim=-1)
                # dists = metric_utils.euclidean_distance(feats, centers) * weight
                # dists = dists.sum(-1).min()
                # scores.append(1/dists.item())

                "cosine similarity"    
                # dists = torch.cosine_similarity(feats, centers).max()
                # scores.append(dists.item())

                "weighted cosine similarity"
                feats = feats.reshape(-1, 4, 128)
                # weight = torch.softmax(weight, dim=-1)
                # dists = torch.cosine_similarity(feats, centers, dim=-1) * weight
                dists = torch.cosine_similarity(feats, centers, dim=-1)
                dists = dists.mean(-1).max()
                score = score.max()
                scores.append(dists.item() * score.item())

                "KL div"
                # dists = torch.kl_div(torch.log_softmax(feats, dim=-1), torch.softmax(centers, dim=-1)).mean(-1)
                # scores.append((1/dists.min().item()))

                logit = torch.softmax(logit, dim=-1).max()
                logits.append(logit)
                if "ImageNet" == dataset_name:
                    with open(os.path.join(save_dir, f"results_{dataset_name}.txt"), "a") as f:
                        f.write(f"{name},{scores[-1]}\n")
                else:
                    with open(os.path.join(save_dir, f"results_{dataset_name}.txt"), "a") as f:
                        f.write(f"{name},{scores[-1]}\n")

    return scores


if __name__ == "__main__":
    args = parse_args()
    inference(args)
