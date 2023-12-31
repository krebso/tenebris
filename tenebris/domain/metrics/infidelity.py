from typing import Any, Callable

import torch

from captum.metrics import infidelity
from torch import Tensor

from tenebris.domain.interfaces.method import ExplainabilityMethod
from tenebris.domain.interfaces.metric import Metric, ReduceStrategy


class Infidelity(Metric):
    name = "Infidelity"
    reduce_strategy = ReduceStrategy.AVERAGE

    def __init__(self, perturbation_fn: Callable[..., Tensor], baselines: int | Tensor) -> None:
        self._perturbation_fn = perturbation_fn
        self._baselines = baselines

    def _compute(self, method: ExplainabilityMethod, input_: Tensor, target: int | Tensor, **kwargs: Any) -> float:
        metric = infidelity(
            forward_func=method.model(),
            perturb_func=self._perturbation_fn,
            inputs=input_,
            attributions=method.attribute(input_, target),  # type: ignore
            baselines=self._baselines,
            target=target,
        )
        return (torch.sum(metric) / len(metric)).item()
