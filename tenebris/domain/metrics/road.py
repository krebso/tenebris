from typing import Type, cast

import numpy

from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy
from tenebris.domain.methods.pytorch_grad_cam import format_target


class ROADMoRF(Metric):
    name = "Remove and Debias - Most Relevant First"
    reduce_strategy = ReduceStrategy.AVERAGE

    def __init__(self, percentile: int = 90, target_output_class: Type = ClassifierOutputTarget) -> None:
        self._percentile = percentile
        self._target_cls = target_output_class
        self._metric = ROADMostRelevantFirst(percentile=self._percentile)

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor) -> float:
        # improve tensor typing
        attr = method.attribute(input_, target)
        assert isinstance(attr, Tensor)
        score = self._metric(
            input_tensor=input_,
            cams=attr.detach().numpy(),
            targets=format_target(target),
            model=method.model(),
        )
        score = cast(numpy.ndarray, score)
        return sum(score) / len(score)
