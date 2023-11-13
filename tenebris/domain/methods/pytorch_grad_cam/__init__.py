from typing import Any, Callable

import numpy as np
import torch

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor
from torch.nn import Module


def format_tensor(f: Callable[..., tuple[Tensor, np.ndarray]]) -> Callable[..., Tensor]:
    def _format_tensor(*args: Any, **kwargs: Any) -> Tensor:
        input_, attr = f(*args, **kwargs)
        return torch.from_numpy(attr).unsqueeze(1).repeat(1, input_.size()[1], 1, 1)

    return _format_tensor


def format_target(target: int | Tensor) -> list[Module]:
    if isinstance(target, int):
        return [ClassifierOutputTarget(target)]
    if target.dim() == 0:
        return [ClassifierOutputTarget(target.item())]
    if target.dim() == 1:
        return [ClassifierOutputTarget(t.item()) for t in target]
    raise ValueError()
