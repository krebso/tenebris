from abc import ABCMeta, abstractmethod

import torch

from torch import Tensor
from torch.nn import Module


class ExplainabilityMethod(metaclass=ABCMeta):
    name: str

    @abstractmethod
    def model(self) -> Module:
        """Return model the method is using"""

    @abstractmethod
    def _attribute_tensor(self, input_: Tensor, target: Tensor) -> Tensor:
        """Generates explanation for model given input and target"""

    def attribute(self, input_: Tensor | tuple[Tensor, ...], target: int | Tensor) -> Tensor:
        if isinstance(input_, tuple):
            assert all((lambda t: isinstance(t, Tensor), input_))
            assert all((lambda t: len(t.size()) == 3, input_))  # C x W x H
            input_ = torch.stack(input_)

        batch_size = input_.size()[0]
        if isinstance(target, int):
            target = torch.stack([torch.tensor(target) for _ in range(batch_size)])

        if (n_target := target.size()[0]) != batch_size:
            assert n_target == 1
            target = torch.stack([target for _ in range(batch_size)])

        return self._attribute_tensor(input_, target)
