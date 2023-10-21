import torch
from torch import Tensor


def jaccard_similarity(t1: Tensor, t2: Tensor) -> float:
    intersection = torch.sum(t1 * t2).float()
    union = torch.sum(t1 + t2 - t1 * t2).float()
    return intersection / union


def make_binary(t: Tensor) -> Tensor:
    # FIXME: this is not very good name
    return (t > 0).float()
