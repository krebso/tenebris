import torch
from captum.attr import DeepLift
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class DeepLiftMethod(ExplainabilityMethod):
    name = "DeepLift"

    def __init__(self, model: Module, baselines: Tensor | None = None):
        self._model = model
        self._explainer = DeepLift(self._model)
        self._baselines = baselines

    def model(self) -> Module:
        return self._model

    def attribute(self, input_: Tensor, target: int) -> Tensor:
        baselines = self._baselines if self._baselines is not None else torch.zeros_like(input_)
        return self._explainer.attribute(input_, target=target, baselines=baselines)
