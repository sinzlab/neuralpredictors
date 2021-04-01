import torch
from .base import ClonedReadout
from .point_pooled import PointPooled2d
from .gaussian import FullGaussian2d
from .factorized import SpatialXFeatureLinear

#################################################################
#### MultiReadout Base Classes
#################################################################

class MultiReadoutBase(torch.nn.ModuleDict):
    _base_readout = None

    def __init__(self, in_shape_dict, n_neurons_dict, clone_readout=False, **kwargs):
        if self._base_readout is None:
            raise ValueError("Attribute _base_readout must be set")
        super().__init__()

        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k

            readout_kwargs = self.prepare_readout_kwargs(i, k, k0, **kwargs)

            if i == 0 or clone_readout is False:
                self.add_module(
                    k,
                    self._base_readout(in_shape=in_shape_dict[k], outdims=n_neurons_dict[k], **readout_kwargs),
                )
                original_readout = k
            elif i > 0 and clone_readout is True:
                self.add_module(k, ClonedReadout(self[original_readout]))

    @staticmethod
    def prepare_readout_kwargs(self, i, k, k0, **kwargs):
        return kwargs

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def initialize(self, mean_activity_dict):
        for k, mu in mean_activity_dict.items():
            self[k].initialize()
            # TODO: Do we want to initialize the biases with mean activity? Or just leave it to the readout to initialize the bias
            if hasattr(self[k], "bias"):
                self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, data_key=None, average=True):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]

        # TODO: Is the default average=True good here?
        return self[data_key].feature_l1(average=average) * self.gamma_readout
        # TODO: change this to -> return self[data_key].regularizer(average=average). Add regularizer method to all readouts


class MultiReadoutSharedParametersBase(MultiReadoutBase):
    def __init__(self, in_shape_dict, n_neurons_dict, clone_readout=False, **kwargs):
        super().__init__(in_shape_dict, n_neurons_dict, clone_readout, **kwargs)

    def prepare_readout_kwargs(self, i, k, k0, **kwargs):
        readout_kwargs = kwargs.copy()

        if 'grid_mean_predictor' in readout_kwargs:
            if readout_kwargs['grid_mean_predictor'] is not None:
                if readout_kwargs['grid_mean_predictor_type'] == 'cortex':
                    readout_kwargs['source_grid'] = readout_kwargs['source_grids'][k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(readout_kwargs['grid_mean_predictor_type']))
                if readout_kwargs['share_transform']:
                    readout_kwargs['shared_transform'] = None if i == 0 else self[k0].mu_transform

            elif readout_kwargs['share_grid']:
                    readout_kwargs['shared_grid'] = {
                        'match_ids': readout_kwargs['shared_match_ids'][k],
                        'shared_grid': None if i == 0 else self[k0].shared_grid
                    }

            del readout_kwargs['share_transform']
            del readout_kwargs['share_grid']
            del readout_kwargs['grid_mean_predictor_type']

        if 'share_features' in readout_kwargs:
            if readout_kwargs['share_features']:
                readout_kwargs['shared_features'] = {
                        'match_ids': readout_kwargs['shared_match_ids'][k],
                        'shared_features': None if i == 0 else self[k0].shared_features
                    }
            else:
                readout_kwargs['shared_features'] = None
            del readout_kwargs['share_features']
        return readout_kwargs

#################################################################
#### Actual MultiReadouts
#################################################################

class MultiplePointPooled2d(MultiReadoutBase):
    _base_readout = PointPooled2d


class MultipleSpatialXFeatureLinear(MultiReadoutBase):
    _base_readout = SpatialXFeatureLinear


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d