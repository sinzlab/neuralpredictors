from torch import nn


class PoissonEncoder(nn.Module):
    def __init__(self, core, readout, shifter=None, elu_offset=0.0):
        """
        An Encoder that wraps the core, readout and optionally a shifter into one model.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            elu_offset (float): Offset value in the final elu non-linearity. Defaults to 0.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
        """
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
