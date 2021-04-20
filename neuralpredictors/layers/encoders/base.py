from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, core, readout, elu_offset):
        super().__init__()
        self.core = core
        self.readout = readout
        self.offset = elu_offset

    def forward(self, x, data_key=None, **kwargs):
        x = self.core(x)
        x = self.readout(x, data_key=data_key)
        return F.elu(x + self.offset) + 1

    def regularizer(self, data_key):
        return self.core.regularizer() + self.readout.regularizer(data_key=data_key)