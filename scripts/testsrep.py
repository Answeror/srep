from __future__ import print_function, division
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import mxnet as mx
from sigr.evaluation import CrossValEvaluation as CV, Exp
from sigr.data import Preprocess, Dataset
from sigr import Context


one_fold_intra_subject_eval = CV(crossval_type='one-fold-intra-subject', batch_size=1000)
intra_session_eval = CV(crossval_type='intra-session', batch_size=1000)

print('NinaPro DB1')
print('===========')

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('ninapro-db1'),
             dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
             Mod=dict(num_gesture=52,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                      params='.cache/srep-ninapro-db1-one-fold-intra-subject-%d/model-0028.params'))],
        folds=np.arange(27),
        windows=np.arange(1, 501),
        balance=True)
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
    print('15 frames (150 ms) majority voting accuracy: %f' % acc[14])

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.accuracies(
        [Exp(dataset=Dataset.from_name('ninapro-db1'), vote=-1,
             dataset_args=dict(preprocess=Preprocess.parse('ninapro-lowpass')),
             Mod=dict(num_gesture=52,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=1, num_semg_col=10, num_filter=64),
                      params='.cache/srep-ninapro-db1-one-fold-intra-subject-%d/model-0028.params'))],
        folds=np.arange(27))
    print('Per-trial majority voting accuracy: %f' % acc.mean())

print('')
print('CapgMyo DB-a')
print('============')

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.vote_accuracy_curves(
        [Exp(dataset=Dataset.from_name('dba'),
             Mod=dict(num_gesture=8,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      params='.cache/srep-dba-one-fold-intra-subject-%d/model-0028.params'))],
        folds=np.arange(18),
        windows=np.arange(1, 1001))
    acc = acc.mean(axis=(0, 1))
    print('Single frame accuracy: %f' % acc[0])
    print('150 frames (150 ms) majority voting accuracy: %f' % acc[149])

with Context(parallel=True, level='DEBUG'):
    acc = one_fold_intra_subject_eval.accuracies(
        [Exp(dataset=Dataset.from_name('dba'), vote=-1,
             Mod=dict(num_gesture=8,
                      context=[mx.gpu(0)],
                      symbol_kargs=dict(dropout=0, num_semg_row=16, num_semg_col=8, num_filter=64),
                      params='.cache/srep-dba-one-fold-intra-subject-%d/model-0028.params'))],
        folds=np.arange(18))
    print('Per-trial majority voting accuracy: %f' % acc.mean())

#  print('')
#  print('# CSL-HDEMG')
#  print('===========')

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.vote_accuracy_curves(
        #  [Exp(dataset=Dataset.from_name('csl'),
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/srep-csl-intra-session-%d/model-0028.params'))],
        #  folds=np.arange(250),
        #  windows=np.arange(1, 2049),
        #  balance=True)
    #  acc = acc.mean(axis=(0, 1))
    #  print('Single frame accuracy: %f' % acc[0])
    #  print('307 frames (150 ms) majority voting accuracy: %f' % acc[306])

#  with Context(parallel=True, level='DEBUG'):
    #  acc = intra_session_eval.accuracies(
        #  [Exp(dataset=Dataset.from_name('csl'), vote=-1,
             #  dataset_args=dict(preprocess=Preprocess.parse('(csl-bandpass,csl-cut,median)')),
             #  Mod=dict(num_gesture=27,
                      #  adabn=True,
                      #  num_adabn_epoch=10,
                      #  context=[mx.gpu(0)],
                      #  symbol_kargs=dict(dropout=0, num_semg_row=24, num_semg_col=7, num_filter=64),
                      #  params='.cache/srep-csl-intra-session-%d/model-0028.params'))],
        #  folds=np.arange(250))
    #  print('Per-trial majority voting accuracy: %f' % acc.mean())
