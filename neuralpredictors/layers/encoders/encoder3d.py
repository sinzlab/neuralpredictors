import torch
from torch import nn


class Encoder3d(nn.Module):
    def __init__(self, core, readout, readout_nonlinearity, elu_xshift, elu_yshift):
        super().__init__()
        self.core = core
        self.readout = readout
        if readout_nonlinearity == "adaptive_elu":
            self.nonlinearity = core.nonlinearities[readout_nonlinearity](elu_xshift, elu_yshift)
        else:
            self.nonlinearity = core.nonlinearities[readout_nonlinearity]()
        self.visualization_dir = None

    def forward(self, x, data_key=None):
        out_core = self.core(x)
        out_core = torch.transpose(out_core, 1, 2)
        # the expected readout is 2d whereas the core can output 3d matrices
        # therefore, the first two dimensions (representing depth and batch size) are flattened and then passed
        # through the readout
        out_core = out_core.reshape(((-1,) + out_core.size()[2:]))

        readout_out = self.readout(out_core)
        out = self.nonlinearity(readout_out)
        return out
