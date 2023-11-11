from abc import ABCMeta, abstractmethod
from enum import Enum

from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod


class ReduceStrategy(Enum):
    ACCURACY = "accuracy"
    AVERAGE = "average"


class Metric(metaclass=ABCMeta):
    name: str
    reduce_strategy: ReduceStrategy

    @abstractmethod
    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor) -> float:
        """Compute the metric for method"""

    def compute(self, methods: list[ExplainabilityMethod], input_: Tensor, target: int | Tensor) -> dict:
        metric = {}

        for method in methods:
            metric[method.name] = self._compute(method, input_, target)

        return metric
