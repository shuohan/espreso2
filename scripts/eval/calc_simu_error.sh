#!/usr/bin/env bash
j

psf_est_dir=$(realpath $(dirname $0)/../..)
simu_dir=~/Code/shuo/utils/lr-simu
proc_dir=~/Code/shuo/utils/image-processing-3d
data_dir=/data
ssim_dir=~/Code/others/ssim_and_psnr_3d


num_epochs=(10000 11000 12000 13000 14000 15000)

for ne in ${num_epochs[@]}; do
    docker run --gpus device=0 --rm \
        -v $psf_est_dir:$psf_est_dir \
        -v $simu_dir:$simu_dir \
        -v $proc_dir:$proc_dir \
        -v $data_dir:$data_dir \
        -v $ssim_dir:$ssim_dir \
        --user $(id -u):$(id -g) \
        -e PYTHONPATH=$psf_est_dir:$proc_dir:$simu_dir:$ssim_dir \
        -w $PWD -t \
        pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime \
        ./calc_simu_error.py ${ne}
done
