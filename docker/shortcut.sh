#!/bin/bash
visdom() {
    docker run -p 8097:8097 -d visdom
}

rl() {
    docker run  \
           -e ROOT_DIR='/workdir' \
           -v $HOME/workdir/rl_playground:/workdir \
           -v $HOME/workdir/pytorch-a2c-ppo-acktr/a2c_ppo_acktr:/workdir/a2c_ppo_acktr \
           -p "8888:8888" \
           -p "5901:5901" \
           -p "6901:6901" \
           --rm \
           -d \
           rl /run_jupyter.sh
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

krl() {
    docker run  \
           -e ROOT_DIR='/workdir' \
           -v $HOME/workdir/rl_playground:/workdir \
           -p "8888:8888" \
           -p "5901:5901" \
           -p "6901:6901" \
           --rm \
           -it \
           rl bash
}

tensorboard() {
    docker run  \
           -e ROOT_DIR='/workdir' \
           -v $PWD:/workdir \
           -p "6006:6006" \
           --rm \
           -d \
           rl tensorboard --logdir=$1
}


