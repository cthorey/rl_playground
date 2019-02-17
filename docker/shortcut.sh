#!/bin/bash
visdom() {
    docker run -p 8097:8097 -d visdom
}

rlgpu() {
    docker run  \
           --runtime=nvidia \
           -e DISPLAY=$DISPLAY \
           --privileged=true \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -e NVIDIA_VISIBLE_DEVICES=${1:-0} \
           -e ROOT_DIR='/workdir' \
           -v $HOME/workdir/rl_playground:/workdir \
           -p "8889:8888" \
           --rm \
           -d \
           rl /run_jupyter.sh
}

rlgpu_run() {
    docker run  \
           --runtime=nvidia \
           -e NVIDIA_VISIBLE_DEVICES=${1:-0} \
           -e ROOT_DIR='/workdir' \
           -v $HOME/workdir/rl_playground:/workdir \
           --rm \
           -d \
           rl $2
}

tensorboard() {
    docker run  \
           -e ROOT_DIR='/workdir' \
           -v $PWD:/workdir \
           -p "6006:6006" \
           --rm \
           -it \
           rl tensorboard --logdir=$1
}


