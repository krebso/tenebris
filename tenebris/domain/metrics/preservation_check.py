from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy
from tenebris.domain.tensors.functions import make_binary


class PreservationCheck(Metric):
    name = "PreservationCheck"
    reduce_strategy = ReduceStrategy.ACCURACY

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor, **kwargs) -> bool:
        model = method.model()
        output = model(input_)
        mask = make_binary(method.attribute(input_, target))
        masked_input = input * mask
        masked_output = model(masked_input)

        return output == masked_output  # FIXME: those are probably not the classes I want to compare
