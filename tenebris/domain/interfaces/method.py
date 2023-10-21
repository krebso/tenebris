from abc import abstractmethod, ABCMeta

from torch import Tensor
from torch.nn import Module


class ExplainabilityMethod(metaclass=ABCMeta):
    name: str

    @abstractmethod
    def model(self) -> Module:
        """
        Returns:
             Model the method is using
        """

    @abstractmethod
    def attribute(
        self, input_: Tensor, target: int | Tensor
    ) -> Tensor:
        """Generates explanation for given model and input

        Args:
            input_: torch Tensor
            target: with respect to what we compute the attribution

        Returns:
            Tensor of explanations
        """
