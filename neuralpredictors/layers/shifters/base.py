from torch import nn


class Shifter(nn.Module):
    """
    Abstract base class for a shifter. It's strongly adviced that the regularizer and initialize methods are implemented by the inheriting class.
    """

    def __repr__(self):
        s = super().__repr__()
        s += " [{} regularizers: ".format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: "gamma" in x, dir(self)):
            ret.append("{} = {}".format(attr, getattr(self, attr)))
        return s + "|".join(ret) + "]\n"

    def regularizer(self):
        """
        Regularizer method to be used during training.
        """
        raise NotImplementedError("regularizer method must be implemented by the inheriting class")

    def initialize(self):
        """
        weight initialization of the torch.parameters
        """
        raise NotImplementedError("initialize method must be implemented by the inheriting class")
