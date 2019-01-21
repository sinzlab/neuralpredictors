import numpy as np
import torch

class Invertible:
    def inv(self, y):
        raise NotImplemented('Subclasses of Invertible must implement an inv method')

class DataTransform:

    def __repr__(self):
        return self.__class__.__name__

    def column_transform(self, label):
        return label

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class MovieTransform(DataTransform):
    pass


class StaticTransform(DataTransform):
    pass


class Subsequence(MovieTransform):
    def __init__(self, frames):
        self.frames = frames

    def __call__(self, x):
        t = x.inputs.shape[1]
        i = np.random.randint(0, t - self.frames)
        return x.__class__(
            **{k: getattr(x, k)[:, i:i + self.frames, ...] if k == 'inputs' else getattr(x, k)[i:i + self.frames, ...]
               for k in x._fields})

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.frames)


class Subsample(MovieTransform, StaticTransform):
    def __init__(self, idx):
        self.idx = idx
        assert np.ndim(self.idx) == 1, 'Dimensionality of index array has to be 1'

    def __call__(self, x):
        return x.__class__(
            **{k: getattr(x, k)[..., self.idx] if k == 'responses' else getattr(x, k) for k in x._fields})

    def __repr__(self):
        return self.__class__.__name__ + '(n={})'.format(len(self.idx))

    def column_transform(self, label):
        return label[self.idx]


class ToTensor(MovieTransform, StaticTransform, Invertible):
    def __init__(self, cuda=False):
        self.cuda = cuda

    def inv(self, y):
        return y.numpy()

    def __call__(self, x):
        return x.__class__(*[torch.from_numpy(elem.astype(np.float32)).cuda()
                             if self.cuda else torch.from_numpy(elem.astype(np.float32)) for elem in x])


class Identity(MovieTransform, StaticTransform, Invertible):
    def __call__(self, x):
        return x

    def inv(self, y):
        return y


