import numpy as np
import torch

from captum.metrics import infidelity_perturb_func_decorator
from skimage.segmentation import morphological_chan_vese, quickshift
from torch import Tensor


def jaccard_similarity(t1: Tensor, t2: Tensor) -> float:
    intersection = torch.sum(t1 * t2).float()
    union = torch.sum(t1 + t2 - t1 * t2).float()
    return intersection.item() / union.item()


def positive_attribution_mask(t: Tensor) -> Tensor:
    return (t > 0).int()


def negative_attribution_mask(t: Tensor) -> Tensor:
    return (t < 0).int()


def bw_superpixels(t: tuple[Tensor] | Tensor, n_iterations: int = 25) -> Tensor:
    if isinstance(t, tuple):
        return torch.stack([bw_superpixels(t_) for t_ in t])

    np_t = t[0][0].detach().numpy()
    superpixels = morphological_chan_vese(np_t, num_iter=n_iterations)
    return Tensor(np.array([[superpixels]])).int()  # 1 * 1 * w * h


def rgb_superpixels(t: Tensor) -> Tensor:
    np_t = np.transpose(t[0].detach().numpy(), (1, 2, 0))  # w * h * c
    superpixels = np.transpose(quickshift(np_t), (2, 0, 1))
    return Tensor(np.array([superpixels])).int()  # 1 * c * w * h


@infidelity_perturb_func_decorator(multipy_by_inputs=False)
def leave_out_perturbation_fn(input_: Tensor, baselines: int = 0) -> Tensor:
    mask = torch.randint(0, 2, input_.size())
    masked_input = input_ * mask
    baseline = (mask == 0).float() * baselines
    return masked_input + baseline
