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

    def predict_mean(self, x, data_key, *args, **kwargs):
        raise NotImplementedError()

    def predict_variance(self, x, data_key, *args, **kwargs):
        raise NotImplementedError()
