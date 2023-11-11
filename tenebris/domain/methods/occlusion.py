from captum.attr import Occlusion
from torch import Tensor
from torch.nn import Module

from tenebris.domain.interfaces.method import ExplainabilityMethod


class OcclusionMethod(ExplainabilityMethod):
    name = "Occlusion"

    def __init__(
        self, model: Module, sliding_window_shapes: tuple, strides: int | tuple, baselines: int | Tensor
    ) -> None:
        self._model = model
        self._explainer = Occlusion(self._model)
        self._sliding_window_shapes = sliding_window_shapes
        self._strides = strides
        self._baselines = baselines

    def model(self) -> Module:
        return self._model

    def _attribute_tensor(self, input_: Tensor, target: int) -> Tensor:
        return self._explainer.attribute(
            inputs=input_,
            sliding_window_shapes=self._sliding_window_shapes,
            strides=self._strides,
            baselines=self._baselines,
            target=target,
        )
