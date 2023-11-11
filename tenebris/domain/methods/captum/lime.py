from typing import Callable, cast

from captum.attr import Lime
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class LimeMethod(ExplainabilityMethod):
    name = "LIME"

    def __init__(self, model: Module, baselines: int | float | Tensor, superpixel_fn: Callable[[Tensor], Tensor]):
        self._model = model
        self._explainer = Lime(self._model)
        self._baselines = baselines
        self._superpixel_fn = superpixel_fn

    def model(self) -> Module:
        return self._model

    def _attribute_tensor(self, input_: Tensor, target: Tensor) -> Tensor:
        return cast(
            Tensor,
            self._explainer.attribute(
                input_,
                target=target,
                baselines=self._baselines,
                feature_mask=self._superpixel_fn(input_),
                n_samples=100,
            ),
        )
