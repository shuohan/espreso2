#!/usr/bin/env bash

spest_dir=$(realpath $(dirname $0)/..)
proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
simu_dir=~/Code/shuo/utils/lr-simu

docker run --rm \
    -v $spest_dir:$spest_dir \
    -v $proc_dir:$proc_dir \
    -v $config_dir:$config_dir \
    -v $trainer_dir:$trainer_dir \
    -v $simu_dir:$simu_dir \
    -e PYTHONPATH=$spest_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
    --user $(id -u):$(id -g) -w $spest_dir -t \
    pytorch-shan sphinx-build docs/source docs/build
