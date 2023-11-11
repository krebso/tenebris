from captum.attr import LRP
from captum.attr._utils.lrp_rules import EpsilonRule
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class LRPEpsilonMethod(ExplainabilityMethod):
    name = "LRP with Epsilon rule"

    def __init__(self, model: Module) -> None:
        self._model = model
        self._explainer = LRP(self._model)

    def _set_propagation_rules(self) -> None:
        for module in self._model.modules():
            module.rule = EpsilonRule()

    def _unset_propagation_rules(self) -> None:
        for module in self._model.modules():
            module.rule = None

    def model(self) -> Module:
        return self._model

    def _attribute_tensor(self, input_: Tensor, target: int) -> Tensor:
        self._set_propagation_rules()
        attribution = self._explainer.attribute(input_, target)
        self._set_propagation_rules()
        return attribution
