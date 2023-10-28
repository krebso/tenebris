from captum.attr import IntegratedGradients
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class IntegratedGradientsMethod(ExplainabilityMethod):
    name = "IntegratedGradients"

    def __init__(self, model: Module, n_steps: int = 25):
        self._model = model
        self._explainer = IntegratedGradients(self._model)
        self._n_steps = n_steps

    def model(self) -> Module:
        return self._model

    def attribute(self, input_: Tensor, target: int) -> Tensor:
        return self._explainer.attribute(input_, target=target, n_steps=self._n_steps).float()
