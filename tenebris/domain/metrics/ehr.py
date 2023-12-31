from typing import Any

from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy
from tenebris.domain.tensors.functions import jaccard_similarity, positive_attribution_mask


class EHR(Metric):
    name = "Effective Heat Ratio"
    reduce_strategy = ReduceStrategy.AVERAGE

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor, **kwargs: Any) -> float:
        attribution = method.attribute(input_, target)
        assert isinstance(attribution, Tensor)
        binary_explanation = positive_attribution_mask(attribution)
        return jaccard_similarity(binary_explanation, kwargs["annotation"])
