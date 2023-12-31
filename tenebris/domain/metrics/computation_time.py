import time

from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy


class ComputationTime(Metric):
    name = "ComputationTime"
    reduce_strategy = ReduceStrategy.AVERAGE

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor) -> float:
        start_time = time.time()
        method.attribute(input_=input_, target=target)
        end_time = time.time()
        return end_time - start_time
