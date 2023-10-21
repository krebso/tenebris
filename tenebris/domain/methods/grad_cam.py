from captum.attr import GuidedGradCam
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class GradCAMMethod(ExplainabilityMethod):
    name = "GuidedGradCam"

    def __init__(self, model: Module, attribution_layer_index: int = -3):
        self._model = model
        self._attribution_layer_index = attribution_layer_index
        self._explainer = GuidedGradCam(self._model, self._model.features[self._attribution_layer_index])

    def model(self) -> Module:
        return self._model

    def attribute(self, input_: Tensor, target: int) -> Tensor:
        return self._explainer.attribute(input_, target)
