FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    clang \
    llvm \
    cmake \
    python3-setuptools \
    libtinfo-dev \
    zlib1g-dev \
    build-essential \
    libedit-dev \
    libc++-dev \
    libxml2-dev  \
    ninja-build 

RUN pip3 install --user numpy decorator attrs tornado psutil xgboost==1.2.0 cloudpickle

ENV TVM_ROOT=/tvm
ENV STONNE_ROOT=/stonne/stonne
ENV MRNA_ROOT=/mrna
 
RUN git clone https://github.com/gicLAB/bifrost bifrost &&\
 git clone https://github.com/gicLAB/mRNA-bifrost mrna &&\
 git clone https://github.com/gicLAB/stonne-bifrost stonne &&\
 git clone --recursive https://github.com/apache/tvm tvm

RUN ls /usr/local/bin

# Build tvm from source
RUN cd /tvm   &&\
 git checkout 68b7e7147fff8960754e4a3a788fd2474749d121 &&\
 git submodule update --init --recursive &&\
 mkdir build &&\
 cp cmake/config.cmake build &&\
 cd build &&\
 echo set\(USE_LLVM \"/usr/bin/llvm-config --link-static\"\) >> config.cmake &&\
 echo set\(HIDE_PRIVATE_SYMBOLS ON\) >> config.cmake &&\
 cmake .. -GNinja &&\
 ninja &&\
 cd ../python &&\
 python3 setup.py install

# Make Bifrost
RUN cd /bifrost/api &&\
 make

# Run Bifrost ops tests
RUN cd /bifrost/bifrost &&\
 python3 setup.py test
