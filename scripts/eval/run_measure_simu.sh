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

prefixes=(/data/oasis3/measure_simu/sub-OAS30016_ses-d0021_acq-mprage_T1w
          /data/oasis3/measure_simu/sub-OAS30032_ses-d3499_acq-mprage_T1w
          /data/oasis3/measure_simu/sub-OAS30048_ses-d3367_acq-mprage_T1w
          /data/oasis3/measure_simu/sub-OAS30064_ses-d0687_acq-mprage_run-02_T1w
          /data/oasis3/measure_simu/sub-OAS30080_ses-d0048_acq-mprage_T1w)
          # /data/oasis3/measure_simu/sub-OAS30096_ses-d0024_acq-mprage_run-01_T1w
          # /data/oasis3/measure_simu/sub-OAS30112_ses-d0259_acq-mprage_T1w
          # /data/oasis3/measure_simu/sub-OAS30127_ses-d0098_acq-mprage_T1w
          # /data/oasis3/measure_simu/sub-OAS30143_ses-d2235_acq-mprage_run-01_T1w
          # /data/oasis3/measure_simu/sub-OAS30160_ses-d0751_acq-mprage_run-01_T1w)

simu_types=(type-gauss_fwhm-2p0_scale-1p0_len-13
            type-gauss_fwhm-3p0_scale-1p0_len-13
            type-gauss_fwhm-4p0_scale-1p0_len-13)

output_dir=/data/spest/ipmi_measure_simu

for prefix in ${prefixes[@]}; do
    for simu_type in ${simu_types[@]}; do
        suboutdir=$output_dir/$(basename ${prefix})/${simu_type}
        if [ -f $suboutdir/kernel/epoch-${ne}.png ]; then
            continue
        fi
        image=${prefix}_${simu_type}.nii
        kernel=$(echo $image | sed "s/\.nii/_kernel.npy/")
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
            ./train.py -d -i $image -o $suboutdir -k $kernel -kl 17 -sw ${sw} \
            -isz 4 -bs ${bs} -e ${ne} -w 0 -lr ${lr} -lrdk ${lrdk[@]} \
            -lrdc ${lrdc[@]} -wd ${wd} -ps ${ps} -ns 1 -ie ${ie} \
            -knc ${knc} -knh ${knh} -knk ${knk}
    done
done | rush -j 3 {}
