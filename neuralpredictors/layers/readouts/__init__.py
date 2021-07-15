from .base import Readout, ClonedReadout
from .point_pooled import PointPooled2d, SpatialTransformerPooled3d
from .gaussian import Gaussian2d, FullGaussian2d, RemappedGaussian2d, DeterministicGaussian2d, Gaussian3d, UltraSparse
from .factorized import SpatialXFeatureLinear, FullFactorized2d, FullSXF
from .attention import AttentionReadout
from .pyramid import PointPyramid2d
from .multi_readout import MultiReadoutBase, MultiReadoutSharedParametersBase


#### MultiReadouts for backwards compatibility
class MultiplePointPooled2d(MultiReadoutBase):
    _base_readout = PointPooled2d


class MultipleFullGaussian2d(MultiReadoutSharedParametersBase):
    _base_readout = FullGaussian2d


### "SpatialXFeatureLinear" and "FullSXF" are now combined in the new "FullFactorized2d" but their respective MultiReadouts are defined here individually for backwards compatibility
class MultipleSpatialXFeatureLinear(MultiReadoutBase):
    _base_readout = FullFactorized2d


class MultipleFullSXF(MultiReadoutSharedParametersBase):
    _base_readout = FullFactorized2d


class MultipleFullFactorized2d(MultiReadoutSharedParametersBase):
    _base_readout = FullFactorized2d
