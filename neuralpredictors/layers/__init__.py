"""
This package contains all the components that are used to construct system identification models.
This includes:
  - cores: Modules used for extracting nonlinearly computed features shared among neurons.
  - readouts: Linear layers added on top of the core output to predict neurons' responses.
  - encoders: Classes which fuse core, readout, shifter (optional), and modulator (optional) into a single model.
  - shifters: Modules used for shifting (i.e. spatially) the features computed via core.
  - modulators: Modules used to incorporate the effect of other variables (e.g. behavioral variables) in response prediction.
  - special layers (e.g. DepthSeparable Convolution) and activation functions.
"""

from .affine import Bias2DLayer, Scale2DLayer
from .conv import DepthSeparableConv2d
