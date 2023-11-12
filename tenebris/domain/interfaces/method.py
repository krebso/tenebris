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

    def _attribute(self, input_: Tensor, target: int | Tensor) -> Tensor:
        batch_size = input_.size()[0]
        if isinstance(target, int):
            target = torch.stack([torch.tensor(target) for _ in range(batch_size)])

        if (n_target := target.size()[0]) != batch_size:
            assert n_target == 1
            target = torch.cat([target for _ in range(batch_size)], dim=0)

        return self._attribute_tensor(input_, target)

    def attribute(self, input_: Tensor | tuple[Tensor, ...], target: int | Tensor) -> Tensor | tuple[Tensor, ...]:
        if isinstance(input_, tuple):
            assert all((isinstance(t, Tensor) for t in input_))
            assert all((len(t.size()) == 4 for t in input_))  # B x C x W x H
            return tuple(self._attribute(t, target) for t in input_)
        return self._attribute(input_, target)
