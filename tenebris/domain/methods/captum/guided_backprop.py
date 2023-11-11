from typing import cast

from captum.attr import GuidedBackprop
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class GuidedBackpropMethod(ExplainabilityMethod):
    name = "GuidedBackprop"

    def __init__(self, model: Module):
        self._model = model
        self._explainer = GuidedBackprop(self._model)

    def model(self) -> Module:
        return self._model

    def _attribute_tensor(self, input_: Tensor, target: Tensor) -> Tensor:
        return cast(Tensor, self._explainer.attribute(input_, target=target))
