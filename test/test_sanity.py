import pytest

from torch import Tensor


@pytest.mark.parametrize(
    "metric_name",
    [
        "road_metric",
        "sensitivity_metric",
        "infidelity_metric",
    ],
)
@pytest.mark.parametrize(
    "method_name",
    [
        "grad_cam",
        "deconvolution",
        "grad_cam_plus_plus",
        "ablation_cam",
    ],
)
def test_sensitivity(
    method_name: str, metric_name: str, vgg_16_batch: list[Tensor], request: pytest.FixtureRequest
) -> None:
    method = request.getfixturevalue(method_name)
    metric = request.getfixturevalue(metric_name)
    input_, target = vgg_16_batch
    metric.compute([method], input_, target)
