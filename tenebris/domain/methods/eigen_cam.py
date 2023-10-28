from typing import Callable

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class EigenCAMMethod(ExplainabilityMethod):
    name = "EigenCAM"

    def __init__(self, model: Module, layer_getter: Callable[..., Module], use_cuda: bool = False):
        self._model = model
        self._explainer = EigenCAM(model=self._model, target_layers=layer_getter(self._model), use_cuda=use_cuda)

    def model(self) -> Module:
        return self._model

    def attribute(self, input_: Tensor, target: int) -> Tensor:
        return Tensor(self._explainer(input_, [ClassifierOutputTarget(target)])).float()
