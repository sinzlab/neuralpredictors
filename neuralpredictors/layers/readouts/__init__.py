from .attention import AttentionReadout
from .base import ClonedReadout, Readout
from .factorized import FullFactorized2d, FullSXF, SpatialXFeatureLinear
from .gaussian import (
    DeterministicGaussian2d,
    FullGaussian2d,
    Gaussian2d,
    Gaussian3d,
    RemappedGaussian2d,
    UltraSparse,
)
from .multi_readout import MultiReadoutBase, MultiReadoutSharedParametersBase
from .point_pooled import PointPooled2d, SpatialTransformerPooled3d
from .pyramid import PointPyramid2d

### Userguide ###

# In order to build your multi-readout, pass the respective readout to your multi-readout base class
# together with your readout kwargs. Use the MultiReadoutSharedParametersBase if you want to share parameters
# between the readouts, otherwise use the MultiReadoutBase. Note that not all readouts support parameter sharing.

# Example:
# standard_multi_pointpooled_readout = MultiReadoutBase(PointPooled2d, **readout_kwargs)
# parameter_sharing_multi_gaussian_readout = MultiReadoutSharedParametersBase(FullGaussian2d, **readout_kwargs)
