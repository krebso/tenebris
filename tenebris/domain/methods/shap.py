import torch
from captum.attr import GradientShap
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class GradientShapMethod(ExplainabilityMethod):
    name = "GradientShap"

    def __init__(self, model: Module, baselines: Tensor | None = None):
        self._model = model
        self._explainer = GradientShap(self._model)
        self._baselines = baselines

    def model(self) -> Module:
        return self._model

    def attribute(self, input_: Tensor, target: int) -> Tensor:
        #  It is recommended that the number of samples in the baselines'
        #  tensors is larger than one.
        baselines = self._baselines if self._baselines is not None else torch.zeros_like(input_ if isinstance(input_, Tensor) else input_[0])
        return self._explainer.attribute(input_, target=target, baselines=baselines)
