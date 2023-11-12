from collections.abc import Callable

import numpy as np
import torch

from pytorch_grad_cam import GradCAMPlusPlus
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.methods.pytorch_grad_cam import format_target, format_tensor


class GradCAMPlusPlusMethod(ExplainabilityMethod):
    name = "GrasCAMPlusPlus"

    def __init__(self, model: Module, layer_getter: Callable[[Module], list[Module]], use_cuda: bool = False) -> None:
        self._model = model
        self._explainer = GradCAMPlusPlus(model=self._model, target_layers=layer_getter(self._model), use_cuda=use_cuda)

    def model(self) -> Module:
        return self._model

    @format_tensor
    def _attribute_tensor(self, input_: Tensor, target: Tensor) -> tuple[Tensor, np.ndarray]:
        with torch.enable_grad():
            return input_, self._explainer(input_, format_target(target))
