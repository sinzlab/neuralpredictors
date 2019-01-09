from attorch.dataset import H5SequenceSet, Invertible
import numpy as np
import torch



class DataTransform:

    def __repr__(self):
        return self.__class__.__name__

    def column_transform(self, label):
        return label


class Subsequence(DataTransform):
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


class Subsample(DataTransform):
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, x):
        return x.__class__(
            **{k: getattr(x, k)[..., self.idx] if k == 'responses' else getattr(x, k) for k in x._fields})

    def __repr__(self):
        return self.__class__.__name__ + '(n={})'.format(len(self.idx))

    def column_transform(self, label):
        return label[self.idx]


class ToTensor(DataTransform, Invertible):
    def __init__(self, cuda=False):
        self.cuda = cuda

    def inv(self, y):
        return y.numpy()

    def __call__(self, x):
        return x.__class__(*[torch.from_numpy(elem.astype(np.float32)).cuda()
                             if self.cuda else torch.from_numpy(elem.astype(np.float32)) for elem in x])


class Identity(DataTransform, Invertible):
    def __call__(self, x):
        return x

    def inv(self, y):
        return y
