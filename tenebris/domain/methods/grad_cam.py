from captum.attr import GuidedGradCam
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class GradCAMMethod(ExplainabilityMethod):
    name = "GuidedGradCam"

    def __init__(self, model: Module, layer_getter=lambda m: m.features[-3]):
        self._model = model
        self._explainer = GuidedGradCam(self._model, layer_getter(self._model))

    def model(self) -> Module:
        return self._model

    def _attribute_tensor(self, input_: Tensor, target: int) -> Tensor:
        return self._explainer.attribute(input_, target)
