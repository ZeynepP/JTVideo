FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
ENV http_proxy="http://firewall.ina.fr:81" https_proxy="http://firewall.ina.fr:81" no_proxy="localhost,127.0.0.1,localaddress,.ina.fr,priv.ina"

MAINTAINER Zeynep Pehlivan
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Install ffmpeg
RUN apt-get -y update
RUN apt-get install -y ffmpeg

# Set working directory
WORKDIR /usr/src/app
#RUN mkdir -p /usr/src/app/vision
#RUN git clone https://github.com/pytorch/vision.git /usr/src/app/vision
#WORKDIR /usr/src/app/vision
#RUN ls
#RUN python setup.py install
## or, for OSX
## MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py develop
## for C++ debugging, please use DEBUG=1
## DEBUG=1 python setup.py develop
#RUN pip install flake8 typing mypy pytest pytest-mock scipy
#WORKDIR /usr/src/app

RUN pip install pytorch-lightning
RUN pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
RUN pip install fiftyone
RUN pip install fiftyone torch torchvision
RUN pip install tensorboard
RUN pip install torchaudio
RUN pip install torchtext
RUN pip install transformers
RUN pip install datasets
RUN pip install saxpy
# for multi nodes
RUN pip install accelerate
# for detr test
RUN pip install timm


# Set the default command to python3.
COPY ./dockertemp.py /usr/src/app
ENTRYPOINT ["python3"]
CMD [ "dockertemp.py", "/usr/src/temp/config.json" ]