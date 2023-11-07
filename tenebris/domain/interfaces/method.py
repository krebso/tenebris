from abc import abstractmethod, ABCMeta

import numpy as np
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
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

    def attribute(self, input_: Tensor | tuple[Tensor, ...], target: Tensor) -> Tensor | tuple[Tensor, ...]:
        if isinstance(input_, tuple):
            # FIXME target dims
            return tuple(self._attribute_tensor(i, target) for i in input_)
        return self._attribute_tensor(input_, target)


class PytorchGradCAMMethod:
    @staticmethod
    def _format_tensor(t: np.ndarray) -> Tensor:
        # add lost channel to BW images
        return torch.from_numpy(t).unsqueeze(1)
