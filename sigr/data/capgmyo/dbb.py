from __future__ import division
from functools import partial
from itertools import product
from logbook import Logger
from . import Dataset as Base
from .. import get_data
from ... import constant


logger = Logger(__name__)


class Dataset(Base):

    name = 'dbb'
    subjects = list(range(2, 21, 2))
    gestures = list(range(1, 9))
    num_session = 2
    sessions = [1, 2]

    def get_universal_inter_session_data(self, fold, batch_size, preprocess, adabn, minibatch, balance_gesture, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(get_data,
                       framerate=self.framerate,
                       root=self.root,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size,
                       num_semg_row=self.num_semg_row,
                       num_semg_col=self.num_semg_col)
        session = fold + 1
        subjects = list(range(1, 11))
        num_subject = 10
        train = load(combos=self.get_combos(product([self.encode_subject_and_session(s, i) for s, i in
                                                     product(subjects, [i for i in self.sessions if i != session])],
                                                    self.gestures, self.trials)),
                     adabn=adabn,
                     mini_batch_size=batch_size // (num_subject * (self.num_session - 1) if minibatch else 1),
                     balance_gesture=balance_gesture,
                     random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                     random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                     random_shift_vertical=kargs.get('random_shift_vertical', 0),
                     shuffle=True)
        logger.debug('Training set loaded')
        val = load(combos=self.get_combos(product([self.encode_subject_and_session(s, session) for s in subjects],
                                                  self.gestures, self.trials)),
                   adabn=adabn,
                   mini_batch_size=batch_size // (num_subject if minibatch else 1),
                   shuffle=False)
        logger.debug('Test set loaded')
        return train, val

    def get_inter_session_data(self, fold, batch_size, preprocess, adabn, minibatch, balance_gesture, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(get_data,
                       framerate=self.framerate,
                       root=self.root,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size,
                       num_semg_row=self.num_semg_row,
                       num_semg_col=self.num_semg_col)
        subject = fold // self.num_session + 1
        session = fold % self.num_session + 1
        train = load(combos=self.get_combos(product([self.encode_subject_and_session(subject, i) for i in self.sessions if i != session],
                                                    self.gestures, self.trials)),
                     adabn=adabn,
                     mini_batch_size=batch_size // (self.num_session - 1 if minibatch else 1),
                     balance_gesture=balance_gesture,
                     random_shift_fill=kargs.get('random_shift_fill', constant.RANDOM_SHIFT_FILL),
                     random_shift_horizontal=kargs.get('random_shift_horizontal', 0),
                     random_shift_vertical=kargs.get('random_shift_vertical', 0),
                     shuffle=True)
        logger.debug('Training set loaded')
        val = load(combos=self.get_combos(product([self.encode_subject_and_session(subject, session)],
                                                  self.gestures, self.trials)),
                   shuffle=False)
        logger.debug('Test set loaded')
        return train, val

    def get_inter_session_val(self, fold, batch_size, preprocess=None, **kargs):
        get_trial = self.get_trial_func(self.gestures, self.trials, preprocess=preprocess)
        load = partial(get_data,
                       framerate=self.framerate,
                       root=self.root,
                       last_batch_handle='pad',
                       get_trial=get_trial,
                       batch_size=batch_size,
                       num_semg_row=self.num_semg_row,
                       num_semg_col=self.num_semg_col)
        subject = fold // self.num_session + 1
        session = fold % self.num_session + 1
        val = load(combos=self.get_combos(product([self.encode_subject_and_session(subject, session)],
                                                  self.gestures, self.trials)),
                   shuffle=False)
        logger.debug('Test set loaded')
        return val

    def encode_subject_and_session(self, subject, session):
        return (subject - 1) * self.num_session + session
