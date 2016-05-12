# Gesture Recognition by Instantaneous Surface EMG Images

This repo contains the code for the experiments in the paper:
Weidong Geng, Yu Du, Wenguang Jin, Wentao Wei, Yu Hu, Jiajun Li. "Gesture recognition by instantaneous surface EMG images." Scientific Reports 6 (2016).

Please see <http://zju-capg.org/myo> for details.

## Requirements

* A CUDA compatible GPU
* Ubuntu 14.04 or any other Linux/Unix that can run Docker
* [Docker](http://docker.io/)
* [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)

## Usage

Following commands will
(1) pull docker image (see `docker/Dockerfile` for details);
(2) train ConvNets on the training sets of NinaPro DB1, CapgMyo DB-a and CSL-HDEMG, respectively;
and (3) test trained ConvNets on the test sets.

```
mkdir .cache
# put NinaPro DB1 in .cache/ninapro-db1
# put CapgMyo DB-a in .cache/dba
# put CSL-HDEMG in .cache/csl
docker pull answeror/sigr:2016-09-21
scripts/trainsrep.sh
scripts/testsrep.sh
```

Training on NinaPro and CapgMyo will take 1 to 2 hours depending on your GPU.
Training on CSL-HDEMG will take several days.
You can accelerate traning and testing by distribute different folds on different GPUs with the `gpu` parameter.

The NinaPro DB1 should be segmented according to the gesture labels and stored in Matlab format as follows.
`.cache/ninapro-db1/data/sss/ggg/sss_ggg_ttt.mat` contains a field `data` (frames x channels) represents the trial `ttt` of gesture `ggg` of subject `sss`.
Numbers are starting from zero. Gesture 0 is the rest posture.
For example, `.cache/ninapro-db1/data/000/001/000_001_000.mat` is the 0th trial of 1st gesture of 0th subject,
and `.cache/ninapro-db1/data/002/003/002_003_004.mat` is the 4th trial of 3th gesture of 2nd subject.
You can download the prepared dataset from <http://zju-capg.org/myo/data/ninapro-db1.zip> or prepare it by yourself.

## License

Licensed under an [GPL v3.0](https://github.com/Answeror/srep/blob/master/LICENSE) license.

## Misc

Thanks DMLC team for their great [MxNet](https://github.com/dmlc/mxnet)!
