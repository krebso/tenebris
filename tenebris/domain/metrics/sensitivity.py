from typing import Any

from captum.metrics import sensitivity_max
from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy


class Sensitivity(Metric):
    name = "Sensitivity"
    reduce_strategy = ReduceStrategy.AVERAGE

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor, **kwargs: Any) -> float:
        return sensitivity_max(
            explanation_func=method.attribute,
            inputs=input_,
            target=target,
        ).item()
