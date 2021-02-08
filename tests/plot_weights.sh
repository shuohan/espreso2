#!/usr/bin/env bash

proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
dirname=$(realpath $PWD/..)

docker run --gpus 1 --rm \
    -v $dirname:$dirname \
    -v $proc_dir:$proc_dir \
    -v $config_dir:$config_dir \
    --env="DISPLAY" \
    --workdir=/app \
    --volume="$PWD":/app \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --user $(id -u):$(id -g) -w $dirname/tests -it \
    -e PYTHONPATH=$dirname:$config_dir:$proc_dir \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ./plot_weights.py
