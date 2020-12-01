#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
simu_dir=~/Code/shuo/utils/lr-simu
data_dir=/data

images=(/data/brainweb/simu/t1_icbm_normal_1mm_pn3_rf0_type-gauss_fwhm-8p0_scale-0p25_len-13.nii)

ns=10000
sw=0
for image in ${images[@]}; do
    fwhm=$(echo $image | sed "s/.*\(fwhm-.*\)_scale.*/\1/")
    scale=$(echo $image | sed "s/.*\(scale-.*\)_len.*/\1/")
    len=$(echo $image | sed "s/.*\(len-.*\)\.nii/\1/")
    outdir=../tests/results/brainweb_${fwhm}_${scale}_${len}_flip
    kernel=$(echo $image | sed "s/\.nii/_kernel.npy/")
    docker run --gpus device=1 --rm \
        -v $psf_est_dir:$psf_est_dir \
        -v $sssrlib_dir:$sssrlib_dir \
        -v $proc_dir:$proc_dir \
        -v $trainer_dir:$trainer_dir \
        -v $simu_dir:$simu_dir \
        -v $data_dir:$data_dir \
        -v $config_dir:$config_dir \
        --user $(id -u):$(id -g) \
        -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
        -w $psf_est_dir/scripts -t \
        pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime \
        ./train.py -i $image -o $outdir -k $kernel -kl 21 -isz 4 -e 10000 \
        -sw ${sw} -wd 1e-2 -lrdk 3,1 3,1 3,1 1,1 1,1 1,1 1,1 1,1 1,1 1,1 \
        -lrdc 64 64 64 64 64 64 64 64 64 -knc 6 -ns ${ns} -ps 16 
        # -lrdc 128 128 128 128 128 128 128 128 128
done # | rush -j 3 {}
