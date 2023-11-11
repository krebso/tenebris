from abc import ABCMeta, abstractmethod
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module


class ExplainabilityMethod(metaclass=ABCMeta):
    name: str

    @abstractmethod
    def model(self) -> Module:
        """
        Returns:
             Model the method is using
        """

    @abstractmethod
    def _attribute_tensor(self, input_: Tensor, target: int | Tensor) -> Tensor:
        """Generates explanation for given model and input

        Args:
            input_: torch Tensor
            target: with respect to what we compute the attribution

        Returns:
            Tensor of explanations
        """

    def attribute(self, input_: Tensor | tuple[Tensor, ...], target: int | Tensor) -> Tensor:
        if isinstance(input_, tuple):
            assert all(map(lambda t: isinstance(t, Tensor), input_))
            assert all(map(lambda t: len(t.size()) == 3, input_))  # C x W x H
            input_ = torch.stack(input_)

        batch_size = input_.size()[0]
        if isinstance(target, int):
            target = torch.stack([torch.tensor(target) for _ in range(batch_size)])

        return self._attribute_tensor(input_, target)


def pytorch_gradcam_format_explanation(f: Callable[..., np.ndarray]) -> Callable[..., Tensor]:
    # add lost channel to BW images
    # assert this is only when it is indeed bw image kekw
    def _format_tensor(*args, **kwargs) -> Tensor:
        return torch.from_numpy(f(*args, **kwargs)).unsqueeze(1)

    return _format_tensor
