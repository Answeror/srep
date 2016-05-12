from __future__ import division
from . import Dataset as Base


class Dataset(Base):

    name = 'dba'
    subjects = list(range(1, 19))
    gestures = list(range(1, 9))
