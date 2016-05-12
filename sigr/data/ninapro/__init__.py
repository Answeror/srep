from __future__ import division
import os
from itertools import product
import numpy as np
import scipy.io as sio
from logbook import Logger
from ... import utils, CACHE
from .. import Dataset as Base, Combo, Trial, SingleSessionMixin


NUM_SEMG_ROW = 1
NUM_SEMG_COL = 10
FRAMERATE = 100
PREPROCESS_KARGS = dict(
    framerate=FRAMERATE,
    num_semg_row=NUM_SEMG_ROW,
    num_semg_col=NUM_SEMG_COL
)

logger = Logger(__name__)


class Dataset(SingleSessionMixin, Base):

    framerate = FRAMERATE
    num_semg_row = NUM_SEMG_ROW
    num_semg_col = NUM_SEMG_COL
    subjects = list(range(27))
    gestures = list(range(53))
    trials = list(range(10))

    def __init__(self, root):
        self.root = root

    def get_one_fold_intra_subject_trials(self):
        return [0, 2, 3, 5, 7, 8, 9], [1, 4, 6]

    def get_trial_func(self, *args, **kargs):
        return GetTrial(*args, **kargs)

    @classmethod
    def parse(cls, text):
        if cls is not Dataset and text == cls.name:
            return cls(root=os.path.join(CACHE, cls.name.split('/')[0], 'data'))


class GetTrial(object):

    def __init__(self, gestures, trials, preprocess=None):
        self.preprocess = preprocess
        self.memo = {}
        self.gesture_and_trials = list(product(gestures, trials))

    def get_path(self, root, combo):
        return os.path.join(
            root,
            '{c.subject:03d}',
            '{c.gesture:03d}',
            '{c.subject:03d}_{c.gesture:03d}_{c.trial:03d}.mat').format(c=combo)

    def __call__(self, root, combo):
        path = self.get_path(root, combo)
        if path not in self.memo:
            logger.debug('Load subject {}', combo.subject)
            paths = [self.get_path(root, Combo(combo.subject, gesture, trial))
                     for gesture, trial in self.gesture_and_trials]
            self.memo.update({path: data for path, data in
                              zip(paths, _get_data(paths, self.preprocess))})
        data = self.memo[path]
        data = data.copy()
        gesture = np.repeat(combo.gesture, len(data))
        subject = np.repeat(combo.subject, len(data))
        return Trial(data=data, gesture=gesture, subject=subject)


@utils.cached
def _get_data(paths, preprocess):
    #  return list(Context.parallel(
        #  jb.delayed(_get_data_aux)(path, preprocess) for path in paths))
    return [_get_data_aux(path, preprocess) for path in paths]


def _get_data_aux(path, preprocess):
    data = sio.loadmat(path)['data'].astype(np.float32)
    if preprocess:
        data = preprocess(data, **PREPROCESS_KARGS)
    return data


from . import db1, db1_g53, db1_g5, db1_g8, db1_g12, caputo, db1_matlab_lowpass
assert db1 and db1_g53 and db1_g5 and db1_g8 and db1_g12 and caputo and db1_matlab_lowpass
