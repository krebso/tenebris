from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric
from tenebris.domain.tensors.functions import make_binary, jaccard_similarity


class EHR(Metric):
    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor, **kwargs) -> float:
        binary_explanation = make_binary(method.attribute(input_, target))
        return jaccard_similarity(binary_explanation, kwargs["annotation"])
