from torch import nn
from torch.nn import functional as F
from ..activations import OffsetElu
import logging
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, core, readout, output_nonlin=None, mean_activity_dict=None):
        super().__init__()
        self.core = core
        self.readout = readout
        self.output_nonlin = output_nonlin
        self.mean_activity_dict = mean_activity_dict
        self.initialize(mean_activity_dict=mean_activity_dict)

    def initialize(self, mean_activity_dict=None, core_kwargs=None, readout_kwargs=None, nonlin_kwargs=None):
        core_kwargs = core_kwargs or {}
        readout_kwargs = readout_kwargs or {}
        nonlin_kwargs = nonlin_kwargs or {}
        self.core.initialize(**core_kwargs)
        self.readout.initialize(mean_activity_dict=mean_activity_dict, **readout_kwargs)
        if hasattr(self.output_nonlin, 'initialize'):
            self.output_nonlin.initialize(**nonlin_kwargs)
        elif nonlin_kwargs:
            logger.warning('The output nonlinarity does not have initialize method but arguments were given.')

    def forward(self, x, data_key=None, **kwargs):
        x = self.core(x)
        x = self.readout(x, data_key=data_key)
        return self.output_nonlin(x)

    def regularizer(self, data_key=None):
        reg = self.core.regularizer() + self.readout.regularizer(data_key=data_key)
        if hasattr(self.output_nonlin, 'regularizer'):
            reg += self.output_nonlin.regularizer()
        return reg

class OffsetEluEncoder(Encoder):
    def __init__(self, core, readout, elu_offset=1.0, mean_activity_dict=None):
        super().__init__(core, readout, OffsetElu(elu_offset), mean_activity_dict)

    def initialize(self, mean_activity_dict=None, core_kwargs=None, readout_kwargs=None, nonlin_kwargs=None):
        # adjust the mean activity that's passed into readout to account for the shift introdcued
        # by the offseted Elu output nonlinearity
        if mean_activity_dict is not None:
            mean_activity_dict = {k: v - self.output_nonlin.offset for k, v in mean_activity_dict.items()}

        super().initialize(mean_activity_dict=mean_activity_dict, core_kwargs=core_kwargs, readout_kwargs=readout_kwargs, nonlin_kwargs=nonlin_kwargs)
        
