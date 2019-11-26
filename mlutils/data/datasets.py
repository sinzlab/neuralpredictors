import h5py
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np
from .transforms import MovieTransform, StaticTransform, Invertible, Subsequence
from scipy.signal import convolve2d

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

    def __getitem__(self, item):
        return self.__getattr__(item)

    def keys(self):
        return self.h5_handle[self.name].keys()

    def __dir__(self):
        return list(self.h5_handle[self.name].keys())


class AttributeTransformer(AttributeHandler):
    def __init__(self, name, h5_handle, transforms):
        super().__init__(name, h5_handle)
        self.transforms = transforms

    def __getattr__(self, item):
        ret = super().__getattr__(item)
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
    def __init__(self, filename, *data_keys, transforms=None, stats_source=None):
        super().__init__(filename, *data_keys, transforms=transforms)
        self.shuffle_dims = {}
        self.stats_source = stats_source if stats_source is not None else 'all'

    @property
    def n_neurons(self):
        return self[0].responses.shape[1]

    @property
    def neurons(self):
        return AttributeTransformer('neurons', self._fid, self.transforms)

    @property
    def img_shape(self):
        return (1,) + self[0].inputs.shape

    def transformed_mean(self, stats_source=None):
        if stats_source is None:
            stats_source = self.stats_source

        tmp = [np.atleast_1d(self.statistics['{}/{}/mean'.format(dk, stats_source)][()])
               for dk in self.data_groups]
        return self.transform(self.data_point(*tmp), exclude=Subsequence)

    def rf_base(self, stats_source='all'):
        N, c, t, w, h = self.img_shape
        t = min(t, 150)
        mean = lambda dk: self.statistics['{}/{}/mean'.format(dk, stats_source)][()]
        d = dict(
            inputs=np.ones((1, c, t, w, h)) * np.array(mean('inputs')),
            eye_position=np.ones((1, t, 1)) * mean('eye_position')[None, None, :],
            behavior=np.ones((1, t, 1)) * mean('behavior')[None, None, :],
            responses=np.ones((1, t, 1)) * mean('responses')[None, None, :]
        )
        return self.transform(self.data_point(*[d[dk] for dk in self.data_groups]), exclude=Subsequence)

    def rf_noise_stim(self, m, t, stats_source='all'):
        """
        Generates a Gaussian white noise stimulus filtered with a 3x3 Gaussian filter
        for the computation of receptive fields. The mean and variance of the Gaussian
        noise are set to the mean and variance of the stimulus ensemble.

        The behvavior, eye movement statistics, and responses are set to their respective means.
        Args:
            m: number of noise samples
            t: length in time

        Returns: tuple of input, behavior, eye, and response

        """
        N, c, _, w, h = self.img_shape
        stat = lambda dk, what: self.statistics['{}/{}/{}'.format(dk, stats_source, what)][()]
        mu, s = stat('inputs', 'mean'), stat('inputs', 'std')
        h_filt = np.float64([
            [1 / 16, 1 / 8, 1 / 16],
            [1 / 8, 1 / 4, 1 / 8],
            [1 / 16, 1 / 8, 1 / 16]]
        )
        noise_input = np.stack([convolve2d(np.random.randn(w, h), h_filt, mode='same')
                                for _ in range(m * t * c)]).reshape((m, c, t, w, h)) * s + mu

        mean_beh = np.ones((m, t, 1)) * stat('behavior', 'mean')[None, None, :]
        mean_eye = np.ones((m, t, 1)) * stat('eye_position', 'mean')[None, None, :]
        mean_resp = np.ones((m, t, 1)) * stat('responses', 'mean')[None, None, :]

        d = dict(
            inputs=noise_input.astype(np.float32),
            eye_position=mean_eye.astype(np.float32),
            behavior=mean_beh.astype(np.float32),
            responses=mean_resp.astype(np.float32)
        )

        return self.transform(self.data_point(*[d[dk] for dk in self.data_groups]), exclude=Subsequence)


    def __getitem__(self, item):
        x = self.data_point(*(np.array(self._fid[g][str(item)]) for g in self.data_groups))

        for tr in self.transforms:
            assert isinstance(tr, MovieTransform)
            x = tr(x)
        return x

    def __repr__(self):
        s =  'MovieSet m={}:\n\t({})'.format(len(self), ', '.join(self.data_groups))
        if self.transforms is not None:
            s += '\n\t[Transforms: ' + '->'.join([repr(tr) for tr in self.transforms]) + ']'
        return s


class H5ArraySet(TransformDataset):
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
            assert isinstance(tr, StaticTransform)
            x = tr(x)
        return x

    def __iter__(self):
        yield from map(self.__getitem__, range(len(self)))

    def __len__(self):
        return self._len

    def __repr__(self):
        return '\n'.join(['Tensor {}: {} '.format(key, self._fid[key].shape)
                          for key in self.data_keys] + ['Transforms: ' + repr(self.transforms)])


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

