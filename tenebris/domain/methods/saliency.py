import torch
from captum.attr import Saliency
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class SaliencyMethod(ExplainabilityMethod):
    name = "Saliency"

    def __init__(self, model: Module):
        self._model = model
        self._explainer = Saliency(self._model)

    def model(self) -> Module:
        return self._model

    def attribute(self, input_: Tensor, target: int) -> Tensor:
        return self._explainer.attribute(input_, target=target)
