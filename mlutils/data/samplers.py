from torch.utils.data import Sampler
import numpy as np

class RepeatsBatchSampler(Sampler):

    def __init__(self, keys, subset_index=None):
        if subset_index is None:
            subset_index = np.arange(len(keys))
        _, inv = np.unique(keys[subset_index], return_inverse=True)
        self.repeat_index = np.unique(inv)
        self.repeat_sets = inv
        self.subset_index = subset_index

    def __iter__(self):
        for u in self.repeat_index:
            yield list(self.subset_index[self.repeat_sets == u])

    def __len__(self):
        return len(self.repeat_index)
