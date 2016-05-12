from __future__ import division
from . import Dataset as Base


class Dataset(Base):

    name = 'ninapro-db1/caputo'
    gestures = list(range(1, 53))

    def get_one_fold_intra_subject_trials(self):
        return [i - 1 for i in [1, 3, 4, 5, 9]], [i - 1 for i in [2, 6, 7, 8, 10]]
