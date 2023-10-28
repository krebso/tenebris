from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import TypeVar

from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod


T = TypeVar("T")


class ReduceStrategy(Enum):
    ACCURACY = "accuracy"
    AVERAGE = "average"


class Metric(metaclass=ABCMeta):
    name: str
    reduce_strategy: ReduceStrategy

    @abstractmethod
    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor) -> T:
        ...

    def compute(self, methods: list[ExplainabilityMethod], input_: Tensor, target: int | Tensor) -> dict:
        """Runs the benchmark and computes the relevant metric for each method

        Args:
            methods: list of methods for which the metric is calculated
            input_: input tensor which is passed to the methods
            target: class with respect to which we compute the attributions

        Returns:
            A dict mapping method name to computed metric value
        """

        metric = {}

        for method in methods:
            metric[method.name] = self._compute(method, input_, target)

        return metric
