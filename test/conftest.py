import pytest
import torch
import torchvision.models

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tenebris.domain.methods.captum.deconvolution import DeconvolutionMethod
from tenebris.domain.methods.captum.grad_cam import GradCAMMethod
from tenebris.domain.methods.pytorch_grad_cam.ablation_cam import AblationCAMMethod
from tenebris.domain.methods.pytorch_grad_cam.grad_cam_plus_plus import GradCAMPlusPlusMethod
from tenebris.domain.metrics.computation_time import ComputationTime
from tenebris.domain.metrics.deletion_check import DeletionCheck
from tenebris.domain.metrics.infidelity import Infidelity
from tenebris.domain.metrics.road import ROADMoRF
from tenebris.domain.metrics.sensitivity import Sensitivity
from tenebris.domain.tensors.functions import leave_out_perturbation_fn


@pytest.fixture(scope="session")
def vgg16_model() -> Module:
    return torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)


@pytest.fixture(scope="session")
def vgg_16_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@pytest.fixture(scope="session")
def cifar_dataset(vgg_16_transforms: transforms.Compose) -> Dataset:
    return torchvision.datasets.CIFAR10(root="test/data", train=False, download=False, transform=vgg_16_transforms)


@pytest.fixture()
def cifar_dataloader(cifar_dataset: Dataset, batch_size: int = 2) -> DataLoader:
    # FIXME we want to have larger batch size!
    return torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


@pytest.fixture()
def vgg_16_batch(cifar_dataloader: DataLoader) -> Tensor:
    return next(iter(cifar_dataloader))


@pytest.fixture()
def grad_cam(vgg16_model: Module) -> GradCAMMethod:
    return GradCAMMethod(model=vgg16_model)


@pytest.fixture()
def deconvolution(vgg16_model: Module) -> DeconvolutionMethod:
    return DeconvolutionMethod(model=vgg16_model)


@pytest.fixture()
def grad_cam_plus_plus(vgg16_model: Module) -> GradCAMPlusPlusMethod:
    return GradCAMPlusPlusMethod(vgg16_model, layer_getter=lambda m: [m.features[-1]])


@pytest.fixture()
def ablation_cam(vgg16_model: Module) -> AblationCAMMethod:
    return AblationCAMMethod(vgg16_model, layer_getter=lambda m: [m.features[-1]])


@pytest.fixture()
def computation_time_metric() -> ComputationTime:
    return ComputationTime()


@pytest.fixture()
def deletion_check_metric() -> DeletionCheck:
    return DeletionCheck()


@pytest.fixture()
def sensitivity_metric() -> Sensitivity:
    return Sensitivity()


@pytest.fixture()
def infidelity_metric() -> Infidelity:
    return Infidelity(perturbation_fn=leave_out_perturbation_fn, baselines=0)


@pytest.fixture()
def road_metric() -> ROADMoRF:
    return ROADMoRF()
