import torch
from torch import nn

from neuralpredictors.constraints import positive


class GradientMixin:
    """
    Test whether the constrain correctly switches off the gradient for tensors and parameters.

    When subclassing, you need to implement a function

    ```
    @staticmethod
    def apply_constrain(w):
        ... # apply the constraint here

    ```

    """

    @staticmethod
    def apply_constrain(w):
        raise NotImplementedError("You need to implement constraint")

    def test_nograd(self):
        w = torch.Tensor([-0.1, 1.0]).requires_grad_(True)
        old_grad_fn = w.grad_fn

        self.apply_constrain(w)

        assert old_grad_fn is w.grad_fn

    def test_nograd_param(self):
        w = nn.Parameter(torch.Tensor([-0.1, 1.0]))

        old_grad_fn = w.grad_fn

        self.apply_constrain(w)

        assert old_grad_fn is w.grad_fn


class TestPositive(GradientMixin):
    tolerance = 1e-8

    @staticmethod
    def apply_constrain(w):
        positive(w)

    def test_positive(self):
        w = torch.Tensor([-1, 1.0])
        self.apply_constrain(w)

        assert torch.all(w >= 0)
