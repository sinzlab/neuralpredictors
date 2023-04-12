from torch.nn import Identity

from neuralpredictors.layers.activations import Elu1

from .base import GeneralizedEncoderBase


class GaussianEncoder(GeneralizedEncoderBase):
    def __init__(self, core, readout, shifter=None, modulator=None, eps=1.e-6):
        nonlinearity_type_list = [Identity(), Elu1()]
        nonlinearity_config_list = [{}, {"inplace": False, "eps": eps}]

        super().__init__(core, readout, nonlinearity_type_list, shifter, modulator, nonlinearity_config_list)

    def predict_mean(self, x, data_key, *args, **kwargs):
        mean, variance = self.forward(x, *args, data_key=data_key, **kwargs)
        return mean

    def predict_variance(self, x, data_key, *args, **kwargs):
        mean, variance = self.forward(x, *args, data_key=data_key, **kwargs)
        return variance


class GammaEncoder(GeneralizedEncoderBase):
    def __init__(self, core, readout, shifter=None, modulator=None, eps=1.e-6):
        nonlinearity_type_list = [Elu1(), Elu1()]
        nonlinearity_config_list = [{"inplace": False, "eps": eps}, {"inplace": False, "eps": eps}]

        super().__init__(core, readout, nonlinearity_type_list, shifter, modulator, nonlinearity_config_list)

    def predict_mean(self, x, data_key, *args, **kwargs):
        concentration, rate = self.forward(x, *args, data_key=data_key, **kwargs)
        return concentration / rate

    def predict_variance(self, x, data_key, *args, **kwargs):
        concentration, rate = self.forward(x, *args, data_key=data_key, **kwargs)
        return concentration / rate**2
