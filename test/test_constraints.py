import torch

from mlutils.constraints import positive


class TestPositive:

    def test_positive(self):
        w = torch.Tensor([-1, 1.])
        positive(w)

        assert torch.all(w >= 0)

    # def test_nograd(self):
    #     w = torch.Tensor([-1, 1.]).requires_grad_(True)
    #     v = torch.Tensor([-1, 1.]).requires_grad_(True)
    #
    #     positive(w)
    #     s1 = w.sum()
    #     s2 = v.sum()
    #     print(s2)
    #     s1.backward()
    #     s2.backward()
    #     print(s2.gradient)
    #     print(s1.grad - s2.grad)
    #     assert torch.all((s1.grad - s2.grad).abs() < 1e-6)