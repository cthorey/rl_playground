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

