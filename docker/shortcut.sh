#!/bin/bash
rl() {
    docker run  \
           -e ROOT_DIR='/workdir' \
           -v $HOME/workdir/rl_playground:/workdir \
           -p "8888:8888" \
           -p "5901:5901" \
           -p "6901:6901" \
           --rm \
           -it \
           rl /run_jupyter.sh
}

rlgpu() {
    docker run  \
           --runtime=nvidia \
           -e NVIDIA_VISIBLE_DEVICES=${1:-0} \
           -e ROOT_DIR='/workdir' \
           -v $HOME/workdir/rl_playground:/workdir \
           -p "8889:8888" \
           --rm \
           -it \
           rlgpu /run_jupyter.sh
}

rlgpu_run() {
    docker run  \
           --runtime=nvidia \
           -e NVIDIA_VISIBLE_DEVICES=${1:-0} \
           -e ROOT_DIR='/workdir' \
           -v $HOME/workdir/rl_playground:/workdir \
           --rm \
           -d \
           rlgpu $2
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
           -it \
           rl tensorboard --logdir=$1
}


