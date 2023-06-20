import torch
from torch import nn


class Encoder(nn.Module):
    """
    Base class for all encoders
    """

    def regularizer(self, data_key=None, reduction="sum", average=None, detach_core=False):
        reg = self.core.regularizer().detach() if detach_core else self.core.regularizer()
        reg = reg + self.readout.regularizer(data_key=data_key, reduction=reduction, average=average)
        if self.shifter:
            reg += self.shifter.regularizer(data_key=data_key)
        if self.modulator:
            reg += self.modulator.regularizer(data_key=data_key)
        return reg

    def predict_mean(self, x, *args, data_key=None, **kwargs):
        raise NotImplementedError()

    def predict_variance(self, x, *args, data_key=None, **kwargs):
        raise NotImplementedError()


class GeneralizedEncoderBase(Encoder):
    def __init__(
        self, core, readout, nonlinearity_type_list, shifter=None, modulator=None, nonlinearity_config_list=None
    ):
        """
        An Encoder that wraps the core, readout and optionally a shifter amd modulator into one model. Can predict any distribution.
        Args:
            core (nn.Module): Core model. Refer to neuralpredictors.layers.cores
            readout (nn.ModuleDict): MultiReadout model. Refer to neuralpredictors.layers.readouts
            nonlinearity_type_list (list of classes/functions): Non-linearity type to use.
            shifter (optional[nn.ModuleDict]): Shifter network. Refer to neuralpredictors.layers.shifters. Defaults to None.
            modulator (optional[nn.ModuleDict]): Modulator network. Modulator networks are not implemented atm (24/06/2021). Defaults to None.
            nonlinearity_config_list (optional[list of dicts]): Non-linearity configuration. Defaults to None.
        """
        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.nonlinearity_type_list = nonlinearity_type_list

        if nonlinearity_config_list is None:
            nonlinearity_config_list = [{}] * len(nonlinearity_type_list)
        self.nonlinearity_config_list = nonlinearity_config_list

    def forward(
        self,
        x,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs
    ):
        # get readout outputs
        x = self.core(x)
        if detach_core:
            x = x.detach()

        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            shift = self.shifter[data_key](pupil_center, trial_idx)

        if "sample" in kwargs:
            x = self.readout(x, data_key=data_key, sample=kwargs["sample"], shift=shift)
        else:
            x = self.readout(x, data_key=data_key, shift=shift)

        # keep batch dimension if only one image was passed
        params = []
        for param in x:
            params.append(param[None, ...] if len(param.shape) == 1 else param)
        x = torch.stack(params)

        if self.modulator:
            x = self.modulator[data_key](x, behavior=behavior)

        assert len(self.nonlinearity_type_list) == len(x) == len(self.nonlinearity_config_list), (
            "Number of non-linearities ({}), number of readout outputs ({}) and, if available, number of non-linearity configs must match. "
            "If you do not wish to restrict a certain readout output with a non-linearity, assign the activation 'Identity' to it."
        )

        output = []
        for i, (nonlinearity, out) in enumerate(zip(self.nonlinearity_type_list, x)):
            output.append(nonlinearity(out, **self.nonlinearity_config_list[i]))

        return tuple(output)
