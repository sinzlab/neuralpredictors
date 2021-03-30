class MultiReadout(Readout, ModuleDict):
    _base_readout = None

    def __init__(self, in_shape, loaders, gamma_readout, clone_readout=False, **kwargs):
        if self._base_readout is None:
            raise ValueError("Attribute _base_readout must be set")

        super().__init__()

        self.in_shape = in_shape
        self.neurons = OrderedDict([(k, loader.dataset.n_neurons) for k, loader in loaders.items()])
        if "positive" in kwargs:
            self._positive = kwargs["positive"]

        self.gamma_readout = gamma_readout  # regularisation strength

        for i, (k, n_neurons) in enumerate(self.neurons.items()):
            if i == 0 or clone_readout is False:
                self.add_module(
                    k,
                    self._base_readout(in_shape=in_shape, outdims=n_neurons, **kwargs),
                )
                original_readout = k
            elif i > 0 and clone_readout is True:
                self.add_module(k, ClonedReadout(self[original_readout], **kwargs))

    def initialize(self, mean_activity_dict):
        for k, mu in mean_activity_dict.items():
            self[k].initialize()
            if hasattr(self[k], "bias"):
                self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_readout

    @property
    def positive(self):
        if hasattr(self, "_positive"):
            return self._positive
        else:
            return False

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value


class MultiplePointPyramid2d(MultiReadout):
    _base_readout = PointPyramid2d


class MultipleGaussian3d(MultiReadout):
    """
    Instantiates multiple instances of Gaussian3d Readouts
    usually used when dealing with different datasets or areas sharing the same core.
    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        loaders (list):  a list of dataset objects
        gamma_readout (float): regularisation term for the readout which is usally set to 0.0 for gaussian3d readout
                               as it contains one dimensional weight

    """

    _base_readout = Gaussian3d

    # Make sure this is not a bug
    def regularizer(self, readout_key):
        return self.gamma_readout


class MultiplePointPooled2d(MultiReadout):
    """
    Instantiates multiple instances of PointPool2d Readouts
    usually used when dealing with more than one dataset sharing the same core.
    """

    _base_readout = PointPooled2d


class MultipleFullGaussian2d(MultiReadout):
    """
    Instantiates multiple instances of FullGaussian2d Readouts
    usually used when dealing with more than one dataset sharing the same core.

    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        loaders (list):  a list of dataloaders
        gamma_readout (float): regularizer for the readout
    """

    _base_readout = FullGaussian2d


class MultipleUltraSparse(MultiReadout):
    """
    This class instantiates multiple instances of UltraSparseReadout
    useful when dealing with multiple datasets
    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        loaders (list):  a list of dataset objects
        gamma_readout (float): regularisation term for the readout which is usally set to 0.0 for UltraSparseReadout readout
                               as it contains one dimensional weight
    """

    _base_readout = UltraSparse
