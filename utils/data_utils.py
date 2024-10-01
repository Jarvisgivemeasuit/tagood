import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from typing import List, Optional

import os
import cv2
import random
import numpy as np
import logging
from tqdm import tqdm
from PIL import Image
import json
cv2.setNumThreads(1)


num_ood_dict = {
    'iNaturalist': [10000, "iNaturalist"],
    'SUN':         [10000, "SUN"],
    'Places':      [10000, "Places"],
    'Textures':    [5640,  "dtd/images"],
    'ImageNet-O':  [2000,  "imagenet-o"],
    'OpenImage':   [17632, "OpenImagesDataset/Images"],
}


def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger


def get_train_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(input_size, (0.6, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def get_val_transforms(input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    pad = 8
    transform = transforms.Compose([
        transforms.Resize((input_size+pad, input_size+pad)),
        transforms.CenterCrop(input_size),
        # transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def generate_attn_mask(x, num_head):
    bs, num_patch, dimen = x.shape
    x = torch.cat([torch.ones(bs, 1, dimen).to(x.device), x], dim=1)
    x_mask = (x != 0).float().reshape(bs, num_patch+1, num_head, dimen//num_head)
    attn_mask = torch.einsum("bihd, bjhd->bhij", x_mask, x_mask)
    attn_mask = (attn_mask == 0).reshape(-1, num_patch+1, num_patch+1) * -1000.0
    return attn_mask


def get_imagenet_one_class_train_loader(data_folder, idx, batch_size, num_workers, input_size, shuffle,
                              get_name=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Get the one class of ImageNet original images training loader.
    '''
    transform = get_val_transforms(input_size, mean, std)
    folder = os.path.join(data_folder, 'train')
    train_set = ImageNetOneClass(folder, idx, transform=transform, get_name=get_name)
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return train_loader


def get_imagenet_one_class_val_loader(data_folder, idx, batch_size, num_workers, input_size, shuffle,
                              get_name=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Get the one class of ImageNet original images validation loader.
    '''
    transform = get_val_transforms(input_size, mean, std)
    folder = os.path.join(data_folder, 'val')
    val_set = ImageNetOneClass(folder, idx, transform=transform, get_name=get_name)
    val_loader = DataLoader(val_set, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return val_loader


def get_ood_loaders(root, batch_size, num_workers, input_size):
    '''
    Get the out-of-distribution original images test loaders.
    '''
    transform   = get_val_transforms(input_size)
    ood_loaders = {}
    def my_collate_fn(batch):
        imgs  = [i[0].unsqueeze(0) for i in batch]
        names = [i[1] for i in batch]
        imgs  = torch.vstack(imgs)
        return imgs, names

    for key in num_ood_dict:
        ood_data         = OODDataset(root, key, transform=transform, get_name=True)
        ood_loader       = DataLoader(ood_data, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                collate_fn=my_collate_fn)
        ood_loaders[key] = ood_loader
        # print(f'{key} loaded with {len(ood_data)} out-of-distribuion test images.')

    return ood_loaders


def get_imagenet_tokens_train_loader_classification(data_folder, 
                                             batch_size, num_workers, shuffle, num_head):
    def my_collate_fn(batch):
    # batch: info,    token_idx,    token_logits, attn,    token,     pred_tag, 
    #        tar_idx, imagenet_tar, word_embed,   tar_tag, file_name
        bs = len(batch)
        image_embed_batch, label_idx_batch = [], []

        for idx in range(bs):
            class_tag   = batch[idx][-2]
            pred_tags   = batch[idx][5]

            image_embed = batch[idx][4]
            label_idx   = batch[idx][6]
            logits      = batch[idx][2]

            image_embed_batch.append(image_embed) # (batch, len(attn > tau), 512)
            label_idx_batch.append  ([label_idx])

        image_embed_batch = torch.nn.utils.rnn.pad_sequence(image_embed_batch, batch_first=True)
        attn_mask = generate_attn_mask(image_embed_batch, num_head)
        label_idx_batch = torch.tensor(label_idx_batch).long()

        return image_embed_batch, attn_mask, label_idx_batch
    
    data_set = ImageNetTokensInfo(data_folder)
    data_loader = DataLoader(data_set, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers, 
                              pin_memory=True, collate_fn=my_collate_fn)
    return data_loader


def get_data_tokens_test_loader_classification(data_folder, batch_size, num_workers):
    def my_collate_fn(batch):
        # batch: info, token_idx, token_logits, attn, token, word_embed, self.tokens[index], labels
        bs = len(batch)
        image_embeddings_batch, attn_batch, name_batch, scores_batch, labels_batch = [], [], [], [], []
        for i in range(bs):

            image_embed = batch[i][-4]
            attn_mask   = batch[i][3]
            file_name   = batch[i][-2]
            labels      = batch[i][-1]
            word_scores = batch[i][2]
            token_idx   = batch[i][1] if isinstance(batch[i][1], list) else [batch[i][1]]

            image_embeddings_batch.append(image_embed)
            name_batch.append(file_name)
            scores_batch.append(word_scores)
            attn_batch.append(attn_mask)
            labels_batch.append(labels)

        # attn_mask_batch = generate_attn_mask(image_embeddings_batch, num_head)
        return image_embeddings_batch, attn_batch, name_batch, scores_batch, labels_batch
        # return image_embeddings_batch, attn_mask_batch, name_batch, scores_batch, labels_batch

    data_set = DatasetTokensForInference(data_folder)
    data_loader = DataLoader(data_set, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers, 
                              pin_memory=True, collate_fn=my_collate_fn)
    return data_loader


def get_ood_token_loaders_classification(root, batch_size, num_workers):
    ood_loaders = {}
    for key in num_ood_dict:
        ood_loaders[key] = get_data_tokens_test_loader_classification(os.path.join(root, key),
                                                               batch_size, num_workers)
    return ood_loaders


def get_imagenet_train_loader_classification(img_root, attn_root, word_embedding_dir, 
                                             batch_size, num_workers, shuffle, num_head):
    transform = get_train_transforms(384)
    def my_collate_fn(batch):
    # batch: img, info, labels
        bs = len(batch)
        image_batch, image_embed_batch, label_idx_batch = [], [], []

        for idx in range(bs):
            image     = batch[idx][0]
            info      = batch[idx][1]
            label_idx = batch[idx][2]
            image_embed = info[3]

            image_batch.append(image)
            image_embed_batch.append(image_embed) # (batch, len(attn > tau), 512)
            label_idx_batch.append([label_idx])

        image_batch = torch.stack(image_batch)
        image_embed_batch = torch.nn.utils.rnn.pad_sequence(image_embed_batch, batch_first=True)
        attn_mask = generate_attn_mask(image_embed_batch, num_head)
        label_idx_batch = torch.tensor(label_idx_batch).long()

        return image_batch, image_embed_batch, attn_mask, label_idx_batch

    data_set = ImageAndAttns(img_root, attn_root, word_embedding_dir, transform, True)
    data_loader = DataLoader(data_set, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers, 
                              pin_memory=True, collate_fn=my_collate_fn)
    return data_loader


def get_imagenet_test_loader_classification(img_root, attn_root, 
                                             batch_size, num_workers, num_head):
    transform = get_val_transforms(384)
    def my_collate_fn(batch):
    # batch: img, info, labels
        bs = len(batch)
        image_batch, image_embed_batch, label_idx_batch = [], [], []

        for idx in range(bs):
            image     = batch[idx][0]
            info      = batch[idx][1]
            label_idx = batch[idx][2]
            image_embed = info[3]

            image_batch.append(image)
            image_embed_batch.append(image_embed) # (batch, len(attn > tau), 512)
            label_idx_batch.append([label_idx])
        
        image_batch = torch.stack(image_batch)
        image_embed_batch = torch.nn.utils.rnn.pad_sequence(image_embed_batch, batch_first=True)
        attn_mask = generate_attn_mask(image_embed_batch, num_head)
        label_idx_batch = torch.tensor(label_idx_batch).long()

        return image_batch, image_embed_batch, attn_mask, label_idx_batch
    
    data_set = ImageAndAttns(img_root, attn_root, transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers, 
                              pin_memory=True, collate_fn=my_collate_fn)
    return data_loader


def get_tokens_loader(data_folder, batch_size, num_workers, shuffle):
    '''
    Get the tokens loader containing:
        - tokens,
        - file names.
    '''
    def my_collate_fn(batch):
        # tokens = [torch.vstack([i[0], torch.zeros(1, i[0].shape[-1])]) for i in batch]
        token_idx = [i[1] for i in batch]
        tokens    = [i[4] for i in batch]
        attns     = [i[3] for i in batch]
        file_name = [i[-1] for i in batch]
        # return token_idx, attns, tokens, file_name
        return token_idx, tokens
    data_set = DatasetTokens(data_folder)
    data_loader = DataLoader(data_set, batch_size=batch_size, 
                              shuffle=shuffle, num_workers=num_workers, 
                              pin_memory=True, collate_fn=my_collate_fn)
    return data_loader


def get_ood_token_loaders(root, batch_size, num_workers):
    ood_loaders = {}

    for key in num_ood_dict:
        ood_loader       = get_tokens_loader(os.path.join(root, key), 
                                             batch_size=batch_size, num_workers=num_workers,shuffle=False)
        ood_loaders[key] = ood_loader
        # print(f'{key} loaded with {len(ood_data)} out-of-distribuion test images.')
    return ood_loaders


class ImageNetOneClass(Dataset):
    def __init__(self, root, idx, transform, get_name=None) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.get_name = False if not get_name else get_name

        with open("texts/ind_name.txt", 'r') as f:
            lines = f.readlines()
            self.class_name = lines[idx].strip().split(" ")[1]

        self.imgs = os.listdir(os.path.join(self.root, self.class_name))

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.class_name, self.imgs[index])).convert("RGB")
        img = self.transform(img)
        if self.get_name:
            return img, self.imgs[index].split(".")[0]
        else:
            return img


class OODDataset(Dataset):
    def __init__(self, ood_root, dataset_name, transform, get_name=None) -> None:
        super().__init__()
        self.ood_root = os.path.join(ood_root, num_ood_dict[dataset_name][1])
        self.dataset_name = dataset_name
        self.transform = transform
        self.get_name = False if not get_name else get_name
        self.img_folders = os.listdir(self.ood_root)
        for i in self.img_folders:
            if i.startswith("."):
                self.img_folders.remove(i)
            if i.endswith(".txt"):
                self.img_folders.remove(i)
        self.imgs = []
        for i in self.img_folders:
            self.list = os.listdir(os.path.join(self.ood_root, i))
            self.imgs += [os.path.join(i, j) for j in self.list]
        for i in self.imgs:
            if not (i.endswith(".jpg") or i.endswith(".png") or i.endswith(".JPEG")):
                self.imgs.remove(i)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        try:
            img = Image.open(os.path.join(self.ood_root, self.imgs[index])).convert("RGB")
        except:
            print(os.path.join(self.ood_root, self.imgs[index]))
        img = self.transform(img)
        if self.get_name:
            return img, self.imgs[index].split(".")[0]
        else:
            return img


class DatasetTokens(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.classes_name = os.listdir(root)
        self.tokens = []
        for name in self.classes_name:
            name_list = os.listdir(os.path.join(self.root, name))
            name_list = [os.path.join(name, i) for i in name_list]
            self.tokens += name_list

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        info      = torch.load(os.path.join(self.root, self.tokens[index]), map_location="cpu")
        file_name = self.tokens[index].split(".")[0]
        tar       = self.classes_name.index(file_name.split("/")[0])

        if len(info) > 3:
            token_idx    = info[0]
            token_logits = info[1]
            attn         = info[2]
            token        = info[3]
        else:
            token_idx    = []
            token_logits = []
            attn         = []
            token        = []

        return info, token_idx, token_logits, attn, token, tar, file_name
    

class ImageNetTokensInfo(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.tokens = []
        self.classes_name = []
        self.classes_tag  = []
        with open("texts/ind_name.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.classes_name.append(line.strip().split(" ")[1])
                self.classes_tag.append(line.split(",")[1].strip())

        ind_tag_file = "texts/ind_tag_list.txt"
        with open(ind_tag_file, "r", encoding="utf-8") as f:
            ind_taglist = [line.strip().split(",") for line in f]
        self.ind_tags  = []
        for info_list in ind_taglist:
            ind_tag  = info_list[1]
            self.ind_tags.append(ind_tag)

        tag_file = "texts/ram_tag_list.txt"
        with open(tag_file, "r", encoding="utf-8") as f:
            self.taglist = [line.strip() for line in f]

        for name in self.classes_name:
            name_list = os.listdir(os.path.join(self.root, name))
            name_list = [os.path.join(name, i) for i in name_list]
            self.tokens += name_list

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        info         = torch.load(os.path.join(self.root, self.tokens[index]), map_location="cpu")
        file_name    = self.tokens[index].split(".")[0]
        tar_idx      = self.classes_name.index(file_name.split("/")[0])
        word_embed   = None
        imagenet_tar = self.classes_tag[tar_idx]
        tar_tag      = self.ind_tags[tar_idx]

        if len(info) > 3:
            token_idx    = info[0]
            token_logits = info[1]
            attn         = info[2]
            token        = info[3]
            if isinstance(token_idx, int):
                token_idx = [token_idx]
            pred_tag     = [self.taglist[i] for i in token_idx]
        else:
            token_idx    = []
            token_logits = []
            attn         = []
            token        = []

        return info, token_idx, token_logits, attn, token, pred_tag, tar_idx, imagenet_tar, word_embed, tar_tag, file_name


class DatasetTokensForInference(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.tokens = []
        self.classes_name = []
        self.classes_tag  = []
        with open("/home/et21-lijl/Documents/multimodalood/texts/ind_name.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.classes_name.append(line.strip().split(" ")[1])
                self.classes_tag.append(line.split(",")[1].strip())

        # ind_tag_file = "/home/et21-lijl/Documents/multimodalood/texts/ind_tag_list.txt"
        ind_tag_file = "/home/et21-lijl/Documents/multimodalood/jupyter/ind_tag_list.txt"
        with open(ind_tag_file, "r", encoding="utf-8") as f:
            ind_taglist = [line.strip().split(",") for line in f]
        self.ind_tags  = []
        for info_list in ind_taglist:
            ind_tag  = info_list[1]
            self.ind_tags.append(ind_tag)

        tag_file = "texts/ram_tag_list.txt"
        with open(tag_file, "r", encoding="utf-8") as f:
            self.taglist = [line.strip() for line in f]

        name_list = os.listdir(os.path.join(self.root))
        for name in name_list:
            file_list = os.listdir(os.path.join(self.root, name))
            file_list = [os.path.join(name, i) for i in file_list]
            self.tokens += file_list

        # self.word_embeddings = torch.load(word_embedding_dir, map_location="cpu")
        # self.word_embeddings = self.word_embeddings.float().detach()

    def token_idx2word_embedding(self, token_idx):
        if isinstance(token_idx, int):
            token_idx = [token_idx]
        if len(token_idx) == 0:
            return []
        # tags = [self.taglist[i] for i in token_idx]
        # indexs = [i for i, x in enumerate(self.ind_tags) if x in tags]
        # word_embedding = self.word_embeddings[indexs]
        # print(word_embedding.shape)
        word_embedding = []
        for idx in token_idx:
            indexs = [i for i, x in enumerate(self.ind_tags) if x == self.taglist[idx]]
            embed = self.word_embeddings[indexs]
            word_embedding.append(embed)
        return word_embedding # list:(len(token_idx), num of tag in ind_tags, 512)

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        info = torch.load(os.path.join(self.root, self.tokens[index]), map_location="cpu")
        if len(info) > 3:
            token_idx    = info[0]
            token_logits = info[1]
            attn         = info[2]
            token        = info[3]
            # word_embed   = self.token_idx2word_embedding(token_idx)
            word_embed   = None
        else:
            token_idx    = []
            token_logits = []
            attn         = []
            token        = []
            word_embed   = []
        
        labels = self.tokens[index].split("/")[0]
        if labels in self.classes_name:
            labels = self.classes_name.index(labels)
        else:
            labels = []

        return info, token_idx, token_logits, attn, token, word_embed, self.tokens[index], labels


class ImageAndAttns(Dataset):
    def __init__(self, image_root, attn_root, transform, train=False) -> None:
        super().__init__()
        self.im_root = image_root
        self.attn_root = attn_root
        self.transform = transform
        self.train = train
        self.folder = "train" if self.train else "val"
        self.class_name = []
        self.img_list   = []
        # self.word_embeddings = torch.load(word_embedding_dir, map_location="cpu")
        # self.word_embeddings = self.word_embeddings.float().detach()
        with open("/home/et21-lijl/Documents/multimodalood/texts/ind_name.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.class_name.append(line.strip().split(" ")[1])

        for name in self.class_name:
            img_list = os.listdir(os.path.join(self.im_root, self.folder, name))
            self.img_list += [os.path.join(name, i) for i in img_list]

        self.attn_list = [i.replace("JPEG", "pt") for i in self.img_list]

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.im_root, 
                                      self.folder,
                                      self.img_list[index])).convert("RGB")
        img = self.transform(img)
        info = torch.load(os.path.join(self.attn_root,
                                       self.attn_list[index]),
                                       map_location="cpu")

        return img, info, self.class_name.index(self.img_list[index].split("/")[0])


class OODImageAndAttns(Dataset):
    def __init__(self, image_root, attn_root, dataset_name, transform) -> None:
        super().__init__()
        self.im_root = os.path.join(image_root, num_ood_dict[dataset_name][1])
        self.attn_root = os.path.join(attn_root, dataset_name)
        self.transform = transform
        self.img_folders = os.listdir(self.im_root)

        self.img_list = []
        for i in self.img_folders:
            if i.endswith(".txt"):
                continue
            elif i.endswith(".directory"):
                self.img_folders.remove(i)
            self.list = os.listdir(os.path.join(self.im_root, i))
            self.img_list += [os.path.join(i, j) for j in self.list if not j.endswith('.directory')]

        self.attn_list = [i.split(".")[0]+".pt" for i in self.img_list]
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.im_root, 
                                      self.img_list[index])).convert("RGB")
        img = self.transform(img)
        info = torch.load(os.path.join(self.attn_root,
                                       self.attn_list[index]),
                                       map_location="cpu")

        return img, info


def get_class_idxs(
    tagging_type: str,
    open_set: bool,
    taglist: List[str]
) -> Optional[List[int]]:
    """Get indices of required categories in the label system."""
    if tagging_type == "ram":
        if not open_set:
            model_taglist_file = "/home/et21-lijl/Documents/multimodalood/texts/ram_tag_list.txt"
            with open(model_taglist_file, "r", encoding="utf-8") as f:
                model_taglist = [line.strip() for line in f]
            return [model_taglist.index(tag) for tag in taglist]
        else:
            return None
    else:  # for tag2text, we directly use tagid instead of text-form of tag.
        # here tagid equals to tag index.
        return [int(tag) for tag in taglist]
    

def load_thresholds(
    threshold: Optional[float],
    threshold_file: Optional[str],
    tagging_type: str,
    open_set: bool,
    class_idxs: List[int],
    num_classes: int,
) -> List[float]:
    """Decide what threshold(s) to use."""
    if not threshold_file and not threshold:  # use default
        if tagging_type == "ram":
            if not open_set:  # use class-wise tuned thresholds
                ram_threshold_file = "/home/et21-lijl/Documents/multimodalood/texts/ram_tag_list_threshold.txt"
                with open(ram_threshold_file, "r", encoding="utf-8") as f:
                    idx2thre = {
                        idx: float(line.strip()) for idx, line in enumerate(f)
                    }
                    return [idx2thre[idx] for idx in class_idxs]
            else:
                return [0.5] * num_classes
        else:
            return [0.68] * num_classes
    elif threshold_file:
        with open(threshold_file, "r", encoding="utf-8") as f:
            thresholds = [float(line.strip()) for line in f]
        assert len(thresholds) == num_classes
        return thresholds
    else:
        return [threshold] * num_classes