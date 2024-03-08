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

    def forward(self, x, data_key=None, pupil_center=None, trial_idx=None, shift=None, detach_core=False, **kwargs):
        out_core = self.core(x)
        if detach_core:
            out_core = out_core.detach()

        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            if shift is None:
                time_points = x.shape[1]
                pupil_center = pupil_center[:, :, -time_points:]
                pupil_center = torch.transpose(pupil_center, 1, 2)
                pupil_center = pupil_center.reshape(((-1,) + pupil_center.size()[2:]))
                shift = self.shifter[data_key](pupil_center, trial_idx)

        out_core = torch.transpose(out_core, 1, 2)
        # the expected readout is 2d whereas the core can output 3d matrices
        # therefore, the first two dimensions (representing depth and batch size) are flattened and then passed
        # through the readout
        out_core = out_core.reshape(((-1,) + out_core.size()[2:]))
        readout_out = self.readout(out_core, data_key=data_key, shift=shift, **kwargs)

        if self.nonlinearity_type == "elu":
            out = self.nonlinearity_fn(readout_out + self.offset) + 1
        else:
            out = self.nonlinearity_fn(readout_out)
        return out
