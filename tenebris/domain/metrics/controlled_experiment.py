from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy
from tenebris.domain.tensors.functions import positive_attribution_mask, jaccard_similarity


class ControlledExperiment(Metric):
    name = "ControlledExperiment"
    reduce_strategy = ReduceStrategy.AVERAGE

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor, **kwargs) -> float:
        annotation = kwargs["annotation"]
        synthetic_input = input_ * annotation
        binary_synthetic_explanation = positive_attribution_mask(method.attribute(synthetic_input, target))

        return jaccard_similarity(annotation, binary_synthetic_explanation)