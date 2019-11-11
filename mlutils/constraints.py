from torch import nn


def constrain_all(self):
    if hasattr(self, 'constrain'):
        self.constrain()

    for c in self.children():
        c.constrain_all()

# extend torch nn.Module to have constrain_all function
nn.Module.constrain_all = constrain_all


# all constrain function takes an optional cache argument
# the cache can be used to store a relatively expensive reusable
# item usable in the constraining. For example, the cache can store the
# binary map of all units that should be constrained.

def positive(weight, cache=None):
    weight.data *= weight.data.ge(0).float()
    return cache


def negative(weight, cache=None):
    weight.data *= weight.data.le(0).float()
    return cache


def positive_except_self(weight, cache=None):
    pos = weight.data.ge(0).float()
    if pos.size()[2] % 2 == 0 or pos.size()[3] % 2 == 0:
        raise ValueError('kernel size must be odd')
    ii, jj = pos.size()[2] // 2, pos.size()[3] // 2
    for i in range(pos.size()[0]):
        pos[i, i, ii, jj] = 1
    weight.data *= pos
    return cache
