from torch import nn


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        """
        A squeeze and excitation block as proposed by https://arxiv.org/abs/1709.01507
        Args:
            in_ch (int): number of input channels
            reduction (int): reduction factor to calculate the output channels.
        """
        super().__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class GlobalAvgPool(nn.Module):
    def __init__(self):
        """
        Helper class used by the SqueezeExcitationBlock
        """
        super().__init__()

    def forward(self, x):
        return x.view(*(x.shape[:-2]), -1).mean(-1)
