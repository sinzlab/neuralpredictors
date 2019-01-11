import h5py
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np

class AttributeTransformer:
    def __init__(self, name, h5_handle, transforms):
        assert name in h5_handle, '{} must be in {}'.format(name, h5_handle)
        self.name = name
        self.h5_handle = h5_handle
        self.transforms = transforms

    def __getattr__(self, item):
        if not item in self.h5_handle[self.name]:
            raise AttributeError('{} is not among the attributes'.format(item))
        else:
            ret = self.h5_handle[self.name][item].value
            if ret.dtype.char == 'S':  # convert bytes to univcode
                ret = ret.astype(str)
            for tr in self.transforms:
                ret = tr.column_transform(ret)
            return ret


class TransformDataset(Dataset):

    def transform(self, x, exclude=None):
        for tr in self.transforms:
            if exclude is None or not isinstance(tr, exclude):
                x = tr(x)
        return x

    def invert(self, x, exclude=None):
        for tr in reversed(filter(lambda tr: not isinstance(tr, exclude), self.transforms)):
            if not isinstance(tr, Invertible):
                raise TypeError('Cannot invert', tr.__class__.__name__)
            else:
                x = tr.inv(x)
        return x

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return '{} m={}:\n\t({})'.format(self.__class__.__name__, len(self), ', '.join(self.data_groups)) \
               + '\n\t[Transforms: ' + '->'.join([repr(tr) for tr in self.transforms]) + ']'


class H5SequenceSet(TransformDataset):
    def __init__(self, filename, *data_groups, transforms=None):
        self._fid = h5py.File(filename, 'r')

        m = None
        for key in data_groups:
            assert key in self._fid, 'Could not find {} in file'.format(key)
            l = len(self._fid[key])
            if m is not None and l != m:
                raise ValueError('groups have different length')
            m = l
        self._len = m

        self.data_groups = data_groups

        self.transforms = transforms or []

        self.data_point = namedtuple('DataPoint', data_groups)

    def __getitem__(self, item):
        x = self.data_point(*(np.array(self._fid[g][str(item)]) for g in self.data_groups))
        for tr in self.transforms:
            x = tr(x)
        return x

    def __getattr__(self, item):
        if item in self._fid:
            item = self._fid[item]
            if isinstance(item, h5py._hl.dataset.Dataset):
                item = item.value
                if item.dtype.char == 'S':  # convert bytes to univcode
                    item = item.astype(str)
                return item
            return item
        else:
            raise AttributeError('Item {} not found in {}'.format(item, self.__class__.__name__))


class MovieSet(H5SequenceSet):
    def __init__(self, filename, *data_keys, transforms=None):
        super().__init__(filename, *data_keys, transforms=transforms)
        self.shuffle_dims = {}

    @property
    def n_neurons(self):
        return self[0].responses.shape[1]

    @property
    def neurons(self):
        return AttributeTransformer('neurons', self._fid, self.transforms)

    @property
    def img_shape(self):
        return (1,) + self[0].inputs.shape


    def __getitem__(self, item):
        x = self.data_point(*(np.array(self._fid[g][
                                           str(item if g not in self.shuffle_dims else self.shuffle_dims[g][item])])
                              for g in self.data_groups))
        for tr in self.transforms:
            x = tr(x)
        return x

    def __repr__(self):
        s =  'MovieSet m={}:\n\t({})'.format(len(self), ', '.join(self.data_groups))
        if self.transforms is not None:
            s += '\n\t[Transforms: ' + '->'.join([repr(tr) for tr in self.transforms]) + ']'
        if len(self.shuffle_dims) > 0:
            s +=  ('\n\t[Shuffled Features: ' + ', '.join(self.shuffle_dims) + ']')
        return s