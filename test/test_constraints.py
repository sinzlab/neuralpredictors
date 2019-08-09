import torch

from mlutils.constraints import positive


class TestPositive:

    tolerance = 1e-8

    def test_positive(self):
        w = torch.Tensor([-1, 1.])
        positive(w)

        assert torch.all(w >= 0)

    def test_nograd(self):
        w = torch.Tensor([-.1, 1.]).requires_grad_(True)
        v = torch.Tensor([-.1, 1.]).requires_grad_(True)

        positive(w)

        s1 = (w * w).sum()
        s2 = (v * v).sum()

        s1.backward()
        s2.backward()
        assert (w.grad[-1] - v.grad[-1]).abs() < self.tolerance
        assert (w.grad[0] - v.grad[0]).abs() > self.tolerance

