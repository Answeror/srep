# Self Made mxnet docker container
FROM lif3line/mxnet:latest
MAINTAINER answeror <answeror@gmail.com>

RUN apt-get install -y python-pip python-scipy
RUN pip install click logbook joblib nose

RUN cd /mxnet && \
  git reset --hard && \
  git checkout master && \
  git pull

RUN cd /mxnet && \
  git checkout 7a485bb && \
  git submodule update && \
  git checkout 887491d src/operator/elementwise_binary_broadcast_op-inl.h && \
  sed -i -e 's/CHECK(ksize_x <= dshape\[3\] && ksize_y <= dshape\[2\])/CHECK(ksize_x <= dshape[3] + 2 * param_.pad[1] \&\& ksize_y <= dshape[2] + 2 * param_.pad[0])/' src/operator/convolution-inl.h && \
  cp make/config.mk . && \
  echo "USE_CUDA=1" >>config.mk && \
  echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk && \
  echo "USE_CUDNN=1" >>config.mk && \
  echo "USE_BLAS=openblas" >>config.mk && \
  make clean && \
  make -j8 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs

ADD elementwise_binary_broadcast_op-inl.h /mxnet/src/operator/elementwise_binary_broadcast_op-inl.h
RUN cd /mxnet && \
  make clean && \
  make -j8 ADD_LDFLAGS=-L/usr/local/cuda/lib64/stubs

# Necessary for newer modules
RUN pip install --upgrade setuptools pip
# Hack to avoid needing python 2.7.9 security features
RUN pip install tornado==4.0
RUN pip install jupyter pandas matplotlib seaborn scikit-learn
RUN mkdir -p -m 700 /root/.jupyter/ && \
  echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888
CMD ["sh", "-c", "jupyter notebook"]

WORKDIR /code
