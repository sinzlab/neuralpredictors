import h5py
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np
from .transforms import MovieTransform, StaticTransform


class AttributeHandler:
    def __init__(self, name, h5_handle):
        assert name in h5_handle, '{} must be in {}'.format(name, h5_handle)
        self.name = name
        self.h5_handle = h5_handle

    def __getattr__(self, item):
        ret = self.h5_handle[self.name][item].value
        if ret.dtype.char == 'S':  # convert bytes to unicode
            ret = ret.astype(str)
        return ret

    def __dir__(self):
        return self.h5_handle[self.name].keys()


class AttributeTransformer(AttributeHandler):
    def __init__(self, name, h5_handle, transforms):
        super().__init__(name, h5_handle)
        self.transforms = transforms

    def __getattr__(self, item):
        ret = super().__getattr__(item)
        for tr in self.transforms:
            ret = tr.column_transform(ret)
        return ret


class Invertible:
    def inv(self, y):
        raise NotImplemented('Subclasses of Invertible must implement an inv method')


class TransformDataset(Dataset):

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
            assert isinstance(tr, MovieTransform)
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
            assert isinstance(tr, MovieTransform)
            x = tr(x)
        return x

    def __repr__(self):
        s =  'MovieSet m={}:\n\t({})'.format(len(self), ', '.join(self.data_groups))
        if self.transforms is not None:
            s += '\n\t[Transforms: ' + '->'.join([repr(tr) for tr in self.transforms]) + ']'
        if len(self.shuffle_dims) > 0:
            s +=  ('\n\t[Shuffled Features: ' + ', '.join(self.shuffle_dims) + ']')
        return s


class H5ArraySet(Dataset):
    def __init__(self, filename, *data_keys, transforms=None):
        self._fid = h5py.File(filename, 'r')
        m = None
        for key in data_keys:
            assert key in self._fid, 'Could not find {} in file'.format(key)
            if m is None:
                m = len(self._fid[key])
            else:
                assert m == len(self._fid[key]), 'Length of datasets do not match'
        self._len = m
        self.data_keys = data_keys

        self.transforms = transforms or []

        self.data_point = namedtuple('DataPoint', data_keys)

    def __getitem__(self, item):
        x = self.data_point(*(self._fid[g][item] for g in self.data_keys))
        for tr in self.transforms:
            x = tr(x)
        return x

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return '\n'.join(['Tensor {}: {} '.format(key, self._fid[key].shape)
                          for key in self.data_keys] + ['Transforms: ' + repr(self.transforms)])

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


class StaticImageSet(H5ArraySet):
    def __init__(self, filename, *data_keys, transforms=None, cache_raw=False, stats_source=None):
        super().__init__(filename, *data_keys, transforms=transforms)
        self.shuffle_dims = {}
        self.cache_raw = cache_raw
        self.last_raw = None
        self.stats_source = stats_source if stats_source is not None else 'all'

    @property
    def n_neurons(self):
        return len(self[0].responses)

    @property
    def neurons(self):
        return AttributeTransformer('neurons', self._fid, self.transforms)

    @property
    def info(self):
        return AttributeHandler('item_info', self._fid)

    @property
    def img_shape(self):
        return (1,) + self[0].images.shape

    def transformed_mean(self, stats_source=None):
        if stats_source is None:
            stats_source = self.stats_source
        tmp = [np.atleast_1d(self.statistics['{}/{}/mean'.format(dk, stats_source)].value)
               for dk in self.data_keys]
        return self.transform(self.data_point(*tmp))

    def __repr__(self):

        return super().__repr__() + \
			('\n\t[Stats source: {}]'.format(self.stats_source) if self.stats_source is not None else '')

