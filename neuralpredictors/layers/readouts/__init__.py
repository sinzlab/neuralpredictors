from .point_pooled import *
from .gaussian import *
from .factorized import *
from .attention import *
from .pyramid import *
from .multi_readout import *


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
