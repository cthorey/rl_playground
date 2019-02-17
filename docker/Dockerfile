FROM nvidia/cuda:9.0-base-ubuntu16.04

# set environment
ENV TERM xterm
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# install python and co
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev \
    git cmake zlib1g-dev \
    xvfb libav-tools \
    xorg-dev libsdl2-dev swig libopenmpi-dev 
RUN pip3 install --upgrade pip

# python packages
RUN git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip3 install -e '.[atari,box2d,classic_control]'
RUN pip3 install pandas seaborn tqdm\
                 jupyter ipython \
                 ipykernel torch torchvision \
                 pybullet sklearn tensorflow \
                 python-box fire tensorboardX \
                 visdom
COPY ./run_jupyter.sh /run_jupyter.sh
COPY ./jupyter_notebook_config.py /root/.jupyter/
WORKDIR /opt
RUN git clone https://github.com/openai/baselines.git && \
    cd baselines && \
    pip install -e .

RUN git clone https://github.com/ikostrikov/pytorch-a2c-ppo-acktr && \
    cd pytorch-a2c-ppo-acktr && \
    pip install -e .

# final
RUN pip install imageio

ENV PYTHONPATH=$PYTHONPATH:/workdir
WORKDIR /workdir
CMD ["bash"]
