from torch.nn import Identity

from neuralpredictors.layers.activations import Elu1

from .base import GeneralizedEncoderBase


class GaussianEncoder(GeneralizedEncoderBase):
    def __init__(self, core, readout, shifter=None, modulator=None, eps=1.0e-6):
        nonlinearity_type_list = [Identity(), Elu1()]
        nonlinearity_config_list = [{}, {"inplace": False, "eps": eps}]

        super().__init__(core, readout, nonlinearity_type_list, shifter, modulator, nonlinearity_config_list)

    def predict_mean(self, x, *args, data_key=None, **kwargs):
        mean, variance = self.forward(x, *args, data_key=data_key, **kwargs)
        return mean

    def predict_variance(self, x, *args, data_key=None, **kwargs):
        mean, variance = self.forward(x, *args, data_key=data_key, **kwargs)
        return variance


class GammaEncoder(GeneralizedEncoderBase):
    def __init__(
        self,
        core,
        readout,
        shifter=None,
        modulator=None,
        eps=1.0e-6,
        max_concentration=None,
        min_rate=None,
        concentration_image_dependent=True,
        rate_image_dependent=True,
    ):
        nonlinearity_type_list = [Elu1(), Elu1()]
        nonlinearity_config_list = [{"inplace": False, "eps": eps}, {"inplace": False, "eps": eps}]

        super().__init__(core, readout, nonlinearity_type_list, shifter, modulator, nonlinearity_config_list)
        self.max_concentration = max_concentration
        self.min_rate = min_rate

        self.concentration_image_dependent = concentration_image_dependent
        self.rate_image_dependent = rate_image_dependent

        if not self.concentration_image_dependent:
            self.concentration_before_nonlinearity = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(1, ro.outdims)) for data_key, ro in self.readout.items()}
            )
        if not self.rate_image_dependent:
            self.rate_before_nonlinearity = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(1, ro.outdims)) for data_key, ro in self.readout.items()}
            )

    def predict_mean(self, x, *args, data_key=None, **kwargs):
        concentration, rate = self.forward(x, *args, data_key=data_key, **kwargs)
        return concentration / rate

    def predict_variance(self, x, *args, data_key=None, **kwargs):
        concentration, rate = self.forward(x, *args, data_key=data_key, **kwargs)
        return concentration / rate**2

    def forward(self, x, *args, data_key=None, **kwargs):
        concentration, rate = super().forward(x, *args, data_key=data_key, **kwargs)

        if not self.concentration_image_dependent:
            concentration = self.nonlinearity_type_list[0](
                self.concentration_before_nonlinearity[data_key], **self.nonlinearity_config_list[0]
            )
        if not self.rate_image_dependent:
            rate = self.nonlinearity_type_list[1](
                self.rate_before_nonlinearity[data_key], **self.nonlinearity_config_list[1]
            )
        if self.min_rate is not None:
            rate = rate.clamp(min=self.min_rate)
        if self.max_concentration is not None:
            concentration = concentration.clamp(max=self.max_concentration)
        return concentration, rate
