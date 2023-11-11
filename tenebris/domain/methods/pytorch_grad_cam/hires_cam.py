from typing import Callable

import numpy as np
import torch

from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.methods.pytorch_grad_cam import format_tensor


class HiResCAMMethod(ExplainabilityMethod):
    name = "HiResCAM"

    def __init__(self, model: Module, layer_getter: Callable[..., Module], use_cuda: bool = False):
        self._model = model
        self._explainer = HiResCAM(model=self._model, target_layers=layer_getter(self._model), use_cuda=use_cuda)

    def model(self) -> Module:
        return self._model

    @format_tensor
    def _attribute_tensor(self, input_: Tensor, target: Tensor) -> np.ndarray:
        if target.dim() == 0:
            targets = [ClassifierOutputTarget(target.item())]
        elif target.dim() == 1:
            targets = [ClassifierOutputTarget(t.item()) for t in target]
        else:
            raise ValueError()
        with torch.enable_grad():
            return self._explainer(input_, targets)