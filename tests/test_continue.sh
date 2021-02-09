#!/usr/bin/env bash


docker run --gpus 1 --rm -v $(realpath $PWD):$(realpath $PWD) --user $(id -u):$(id -g) -w $(realpath $PWD) -it \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ./test_continue.py
