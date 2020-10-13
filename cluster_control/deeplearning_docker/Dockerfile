FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 

#RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV APT_INTALL="apt-get install -y --no-install-recommends"
RUN apt-get update && $APT_INTALL \
         build-essential \
         cmake \
         git \
         openssh-client \
         curl \
         vim-gtk \
         tmux \
         unzip \
         ca-certificates \
         libjpeg-dev \
         libopenexr-dev \
         libpng-dev \
         sudo
         

         

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
ENV CUDNN_VERSION 7.3.1.20
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && $APT_INTALL --allow-unauthenticated \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0  \
                  libnccl2=2.3.5-2+cuda9.0 \
         libnccl-dev=2.3.5-2+cuda9.0 \
         libcupti-dev
         
ENV PYTHON_LIBS="numpy matplotlib h5py scipy scikit-image Pillow tensorboardX torch torchvision tensorflow-gpu"

ENV PIP3I="python3 -m pip install  --upgrade "
ENV PIP2I="python -m pip install  --upgrade "

RUN $APT_INTALL \
    python3 python3-pip python3-dev python3-tk python3-pil.imagetk
RUN $PIP3I --upgrade pip
RUN $PIP3I setuptools
RUN $PIP3I $PYTHON_LIBS
RUN curl -L https://github.com/jamesbowman/openexrpython/zipball/f6fb5bc8cb79744029067cdfb16cc3db9f8cca9f/ -o openexrpython.zip && unzip  openexrpython.zip -d openexrpython && rm openexrpython.zip && cd openexrpython && python3 -m  easy_install -U openexr && cd .. && rm -r openexrpython
 

RUN $APT_INTALL \
    python python-pip python-dev python-tk
RUN $PIP2I --upgrade pip
RUN $PIP2I setuptools
RUN $PIP2I $PYTHON_LIBS
RUN $PIP2I OpenExr

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN mkdir /hooks
COPY startup.sh /hooks/startup.sh
COPY stop.sh /hooks/stop.sh
