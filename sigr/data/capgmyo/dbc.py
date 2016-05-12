from __future__ import division
from . import Dataset as Base


class Dataset(Base):

    name = 'dbc'
    subjects = list(range(1, 11))
    gestures = list(range(1, 13))
