from torch import nn


class PoissonEncoder(nn.Module):
    def __init__(self, core, readout, elu_offset=0, shifter=None):
        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.offset = elu_offset

    def forward(
        self,
        inputs,
        targets=None,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs
    ):
        x = self.core(inputs)
        if detach_core:
            x = x.detach()

        if self.shifter is not None:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            shift = self.shifter[data_key](pupil_center, trial_idx)

        x = self.readout(x, data_key=data_key, shift=shift, **kwargs)
        return nn.functional.elu(x + self.offset) + 1
