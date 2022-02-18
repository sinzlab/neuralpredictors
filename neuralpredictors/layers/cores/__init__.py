"""
For most of our neural prediction models, we use a core and readout architecture.
In this module, all the cores can be found.
The core's task is to encode the input into (hidden) representations, which are shared across all neurons.
These representations will be passed on to the readout to calculate the desired output.

Our cores are usually CNNs. The core module is using special architectures submodules from ..layers (such as
attention_conv which uses self attention instead a conv2d layer).

All core classes must have a `regularizer` and a `forward` method.
"""

from .conv2d import (
    RotationEquivariant2dCore,
    SE2dCore,
    Stacked2dCore,
    TransferLearningCore,
)
