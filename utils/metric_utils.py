import torch
import numpy as np

from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score


class Accuracy:
    def __init__(self, eps=1e-7):
        self.num_correct = 0
        self.num_top5 = 0
        self.num_instance = 0

        self.eps = eps

    def update(self, pred, target):
        # _, ind = torch.sort(pred)
        # ind = ind[:, -5:]
        if len(pred.shape) > 1:
            pred = torch.argmax(pred, dim=1)
        else:
            pred = pred > 0.5
        # self.num_top5 += (target.unsqueeze(-1).expand_as(ind) == ind).sum().item()
        self.num_correct += (pred == target).sum().item()
        self.num_instance += target.shape[0]

    def get_top1(self):
        return self.num_correct / (self.num_instance + self.eps)

    # def get_top5(self):
    #     return self.num_top5 / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0
        self.num_top5 = 0


class KL_Accuracy:
    def __init__(self, eps=1e-7):
        self.num_correct = 0
        self.num_top5 = 0
        self.num_instance = 0

        self.eps = eps

    def update(self, pred, target):
        _, ind = torch.sort(pred)
        ind = ind[:, :5]
        pred = torch.argmin(pred, dim=1)

        self.num_top5 += (target.unsqueeze(-1).expand_as(ind) == ind).sum().item()
        self.num_correct += (pred == target).sum().item()
        self.num_instance += target.shape[0]

    def get_top1(self):
        return self.num_correct / (self.num_instance + self.eps)

    def get_top5(self):
        return self.num_top5 / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0
        self.num_top5 = 0


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def minkowski_distance(p, q, p_value):
    """
    计算闵可夫斯基距离
    :param p: 第一个点的坐标（张量）
    :param q: 第二个点的坐标（张量）
    :param p_value: 闵可夫斯基距离中的 p 参数
    :return: 闵可夫斯基距离
    """
    distance = torch.sum(torch.abs(p - q) ** p_value, dim=1) ** (1/p_value)
    return distance


def euclidean_distance(p, q):
    """
    计算欧氏距离
    :param p: 第一个点的坐标（张量）
    :param q: 第二个点的坐标（张量）
    :return: 欧氏距离
    """
    distance = torch.sqrt(torch.sum((p - q) ** 2, dim=-1))
    return distance


def manhattan_distance(p, q):
    """
    计算曼哈顿距离
    :param p: 第一个点的坐标（张量）
    :param q: 第二个点的坐标（张量）
    :return: 曼哈顿距离
    """
    distance = torch.sum(torch.abs(p - q), dim=1)
    return distance


def get_results(res_in, res_out):
    print(f"ind results:{len(res_in)}, ood_results:{len(res_out)}")
    # tar_in, tar_out = np.zeros(len(res_in)), np.ones(len(res_out))
    tar_in, tar_out = np.ones(len(res_in)), np.zeros(len(res_out))
    res, tar        = [], []

    res.extend(res_in)
    res.extend(res_out)
    tar.extend(tar_in.tolist())
    tar.extend(tar_out.tolist())
    
    auroc = roc_auc_score(tar, res)
    fpr95 = calc_fpr(res, tar)
    aupr = average_precision_score(tar, res)
    return auroc, fpr95, aupr


def calc_fpr(scores, trues):
    tpr95=0.95
    fpr, tpr, thresholds = roc_curve(trues, scores)
    fpr0=0
    tpr0=0
    for i,(fpr1,tpr1) in enumerate(zip(fpr,tpr)):
        if tpr1>=tpr95:
            break
        fpr0=fpr1
        tpr0=tpr1
    fpr95 = ((tpr95-tpr0)*fpr1 + (tpr1-tpr95)*fpr0) / (tpr1-tpr0)
    return fpr95