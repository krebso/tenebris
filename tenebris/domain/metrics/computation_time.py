import time
from datetime import timedelta

from torch import Tensor

from tenebris.domain.interfaces.metric import Metric
from tenebris.domain.interfaces.method import ExplainabilityMethod


class ComputationTimeMetric(Metric):
    name = "ComputationTimeMetric"

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor, **kwargs) -> timedelta:
        start_time = time.time()
        method.attribute(input_=input_, target=target)
        end_time = time.time()
        return end_time - start_time
