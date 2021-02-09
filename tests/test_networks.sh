#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)
config_dir=~/Code/shuo/utils/singleton-config

docker run --gpus device=1 --rm -v $dir:$dir -v $config_dir:$config_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$config_dir -w $dir/tests -t \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ./test_networks.py
