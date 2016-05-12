FROM nvidia/cuda:7.5-cudnn5-devel
MAINTAINER answeror <answeror@gmail.com>

RUN echo "deb http://mirrors.zju.edu.cn/ubuntu/ trusty main restricted universe multiverse" > /etc/apt/sources.list && \
  echo "deb http://mirrors.zju.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://mirrors.zju.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://mirrors.zju.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb http://mirrors.zju.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://mirrors.zju.edu.cn/ubuntu/ trusty main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://mirrors.zju.edu.cn/ubuntu/ trusty-security main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://mirrors.zju.edu.cn/ubuntu/ trusty-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://mirrors.zju.edu.cn/ubuntu/ trusty-proposed main restricted universe multiverse" >> /etc/apt/sources.list && \
  echo "deb-src http://mirrors.zju.edu.cn/ubuntu/ trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
  apt-get -qqy update

# mxnet
RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  libopenblas-dev \
  libopencv-dev \
  python-numpy \
  wget \
  unzip
RUN git clone --recursive https://github.com/dmlc/mxnet/ && cd mxnet && \
  git checkout f2684a6 && \
  sed -i -e 's/CHECK(ksize_x <= dshape\[3\] && ksize_y <= dshape\[2\])/CHECK(ksize_x <= dshape[3] + 2 * param_.pad[1] \&\& ksize_y <= dshape[2] + 2 * param_.pad[0])/' src/operator/convolution-inl.h && \
  cp make/config.mk . && \
  echo "USE_CUDA=1" >>config.mk && \
  echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk && \
  echo "USE_CUDNN=1" >>config.mk && \
  echo "USE_BLAS=openblas" >>config.mk && \
  make -j8 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH 

ENV PYTHONPATH /mxnet/python
