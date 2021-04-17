"""
This package contains shifter-networks. These are used to modify the neurons' readout position
based on the eye position.

Usually, the input to the shifters are the pupil centers of both eyes. Given that input,
a horizontal and vertical shift of the readout position is returned by the shifter.
"""
from .base import Shifter
from .mlp import MLP, MLPShifter
from .static_affine import StaticAffine2d, StaticAffine2dShifter
