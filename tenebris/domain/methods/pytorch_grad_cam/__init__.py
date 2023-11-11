from typing import Any, Callable

import numpy as np
import torch

from torch import Tensor


def format_tensor(f: Callable[..., np.ndarray]) -> Callable[..., Tensor]:
    def _format_tensor(*args: Any, **kwargs: Any) -> Tensor:
        return torch.from_numpy(f(*args, **kwargs)).unsqueeze(1)

    return _format_tensor
