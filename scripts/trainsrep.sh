#!/usr/bin/env bash

# Recognition of 8 gestures in CapgMyo DB-a
scripts/runsrep python -m sigr.app exp --log log --snapshot model \
  --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
  --root .cache/srep-dba-universal-one-fold-intra-subject \
  --num-semg-row 16 --num-semg-col 8 \
  --batch-size 1000 --decay-all --dataset dba \
  --num-filter 64 \
  crossval --crossval-type universal-one-fold-intra-subject --fold 0
for i in $(seq 0 17 | shuf); do
  scripts/runsrep python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/srep-dba-one-fold-intra-subject-$i \
    --num-semg-row 16 --num-semg-col 8 \
    --batch-size 1000 --decay-all --dataset dba \
    --num-filter 64 \
    --params .cache/srep-dba-universal-one-fold-intra-subject/model-0028.params \
    crossval --crossval-type one-fold-intra-subject --fold $i
done

# Recognition of 52 gestures in NinaPro DB1
scripts/runsrep python -m sigr.app exp --log log --snapshot model \
  --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
  --root .cache/srep-ninapro-db1-universal-one-fold-intra-subject \
  --num-semg-row 1 --num-semg-col 10 \
  --batch-size 1000 --decay-all --dataset ninapro-db1 \
  --num-filter 64 \
  --balance-gesture 1 \
  --preprocess 'ninapro-lowpass' \
  crossval --crossval-type universal-one-fold-intra-subject --fold 0
for i in $(seq 0 26 | shuf); do
  scripts/runsrep python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/srep-ninapro-db1-one-fold-intra-subject-$i \
    --num-semg-row 1 --num-semg-col 10 \
    --batch-size 1000 --decay-all --dataset ninapro-db1 \
    --num-filter 64 \
    --params .cache/srep-ninapro-db1-universal-one-fold-intra-subject/model-0028.params \
    --balance-gesture 1 \
    --preprocess 'ninapro-lowpass' \
    crossval --crossval-type one-fold-intra-subject --fold $i
done

# Recognition of 27 gestures in CSL-HDEMG
for i in $(seq 0 24 | shuf); do
  scripts/runsrep python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/srep-csl-universal-intra-session-$i \
    --num-semg-row 24 --num-semg-col 7 \
    --batch-size 2500 --decay-all --adabn --minibatch --dataset csl \
    --preprocess '(csl-bandpass,csl-cut,downsample-5,median)' \
    --balance-gesture 1 \
    --num-filter 64 \
    crossval --crossval-type universal-intra-session --fold $i
done
for i in $(seq 0 249 | shuf); do
  scripts/runsrep python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/srep-csl-intra-session-$i \
    --num-semg-row 24 --num-semg-col 7 \
    --batch-size 1000 --decay-all --adabn --num-adabn-epoch 10 --dataset csl \
    --preprocess '(csl-bandpass,csl-cut,median)' \
    --balance-gesture 1 \
    --params .cache/srep-csl-universal-intra-session-$(($i % 10))/model-0028.params \
    --num-filter 64 \
    crossval --crossval-type intra-session --fold $i
done
