from typing import Any

import torch

from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy
from tenebris.domain.tensors.functions import positive_attribution_mask


class DeletionCheck(Metric):
    name = "DeletionCheck"
    reduce_strategy = ReduceStrategy.ACCURACY

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor, **kwargs: Any) -> bool:
        model = method.model()

        output = model(input_)
        _, output_class = torch.max(output, dim=1)

        attribution = method.attribute(input_, target)
        assert isinstance(attribution, Tensor)
        mask = positive_attribution_mask(attribution)
        masked_input = input_ * (~mask)

        masked_output = model(masked_input)
        _, masked_output_class = torch.max(masked_output, dim=1)

        return output_class.item() != masked_output_class.item()
