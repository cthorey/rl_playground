FROM consol/ubuntu-xfce-vnc

USER 0

# set the workdir
WORKDIR /tmp

# set environment
ENV TERM xterm
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# install python and co
RUN apt-get update
RUN apt-get install -y python3-pip python3-dev
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade tensorflow
RUN apt-get update --fix-missing
RUN apt-get install -y git
RUN apt-get update
RUN apt-get install -y cmake zlib1g-dev xvfb libav-tools xorg-dev libsdl2-dev swig

RUN git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip install -e '.[atari,box2d,classic_control]'
RUN pip3 install pandas seaborn
RUN pip3 install tqdm

RUN pip3 install jupyter
RUN pip3 install ipython ipykernel
COPY ./run_jupyter.sh /run_jupyter.sh
COPY ./jupyter_notebook_config.py /headless/.jupyter/

RUN pip3 install torch torchvision

#open ai baseline
WORKDIR /opt
RUN apt-get update &&\
    apt-get install -y libopenmpi-dev

RUN git clone https://github.com/openai/baselines.git && \
    cd baselines && \
    pip install -e .

# other package
RUN pip3 install sklearn python-box fire tqdm tensorboardX

#robotschool
WORKDIR /opt
RUN git clone https://github.com/openai/roboschool
ENV ROBOSCHOOL_PATH=/opt/roboschool
RUN apt-get update && \
    apt install -y cmake ffmpeg pkg-config qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.5-dev libboost-python-dev libtinyxml-dev
RUN git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision && \
    mkdir bullet3/build && \
    cd    bullet3/build && \
    cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF .. && \
    make -j4 && \
    make install
RUN pip3 install -e $ROBOSCHOOL_PATH

RUN pip3 install imageio
# mujoco

ENV VNC_PW=rl
ENV VNC_RESOLUTION=1440x900
ENV PYTHONPATH=$PYTHONPATH:/workdir
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/headless/.mujoco/mjpro150/bin
WORKDIR /workdir

ENTRYPOINT ["/dockerstartup/vnc_startup.sh"]
CMD ["--wait"]
