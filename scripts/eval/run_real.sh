#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/../..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
simu_dir=~/Code/shuo/utils/lr-simu
data_dir=/data

lr=2e-4
bs=64
ne=15000
lrdk=(3,1 3,1 3,1 1,1 1,1)
lrdc=(64 64 64 64)
sw=1
wd=5e-2
sw_str=$(echo $sw | sed "s/\./p/")
lrdk_str=$(echo ${lrdk[@]} | sed "s/ /-/g")
lrdc_str=$(echo ${lrdc[@]} | sed "s/ /-/g")
ps=16
ie=100

knk=3
knc=3
knh=256

images=(/data/oasis3/data/sub-OAS30016_ses-d0021_acq-TSE_T2w.nii.gz
        /data/oasis3/data/sub-OAS30032_ses-d3499_acq-TSE_T2w.nii.gz
        /data/oasis3/data/sub-OAS30048_ses-d3367_acq-TSE_T2w.nii.gz
        /data/oasis3/data/sub-OAS30064_ses-d0687_acq-TSE_T2w.nii.gz
        /data/oasis3/data/sub-OAS30080_ses-d0048_acq-TSE_T2w.nii.gz)

output_dir=/data/spest/ipmi_esimate_sp_of_real_data

for image in ${images[@]}; do
    suboutdir=$output_dir/$(basename ${image} | sed "s/\.nii\.gz$//")

    if [ -f $suboutdir/kernel/epoch-${ne}.png ]; then
        continue
    fi

    echo docker run --gpus device=0 --rm \
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
        ./train.py -i $image -o $suboutdir -kl 17 -sw ${sw} \
        -isz 4 -bs ${bs} -e ${ne} -w 0 -lr ${lr} -lrdk ${lrdk[@]} \
        -lrdc ${lrdc[@]} -wd ${wd} -ps ${ps} -ns 1 -ie ${ie} \
        -knc ${knc} -knh ${knh} -knk ${knk}

done | rush -j 5 {}
