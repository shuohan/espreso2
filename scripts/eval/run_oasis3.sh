#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/../..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
simu_dir=~/Code/shuo/utils/lr-simu
data_dir=/data

# images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p125_len-13.nii)
# images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-3p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-5p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-9p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-3p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-5p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-9p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-3p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-5p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-9p0_scale-0p125_len-13.nii)
# images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p125_len-13.nii)

images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p5_len-13.nii
        /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p25_len-13.nii
        /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p125_len-13.nii
        /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-3p0_scale-0p5_len-13.nii
        /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-5p0_scale-0p25_len-13.nii
        /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-9p0_scale-0p125_len-13.nii)

lr=2e-4
bs=64
ne=20000
# lrdk=(3,1 3,1 3,1 1,1 1,1 1,1 1,1 1,1 1,1 1,1)
# lrdc=(64 64 64 64 64 64 64 64 64)
lrdk=(3,1 3,1 3,1 1,1 1,1)
lrdc=(64 64 64 64)
sw=1
wd=2e-2
sw_str=$(echo $sw | sed "s/\./p/")
lrdk_str=$(echo ${lrdk[@]} | sed "s/ /-/g")
lrdc_str=$(echo ${lrdc[@]} | sed "s/ /-/g")
ps=16
ie=200

knk=2
knc=3
knh=256

for image in ${images[@]}; do
    fwhm=$(echo $image | sed "s/.*\(fwhm-.*\)_scale.*/\1/")
    scale=$(echo $image | sed "s/.*\(scale-.*\)_len.*/\1/")
    kernel=$(echo $image | sed "s/.*\(type-.*\)_fw.*/\1/")
    len=$(echo $image | sed "s/.*\(len-.*\)\.nii/\1/")
    outdir=/data/spest/results/simu-oasis3_lr-${lr}_bs-${bs}_ne-${ne}_ie-${ie}_sw-${sw_str}_wd-${wd}_ps-${ps}_lrdk-${lrdk_str}_lrdc-${lrdc_str}_knc-${knc}_knh-${knh}_knk-${knk}_orth_clip/${kernel}_${fwhm}_${scale}_${len}
    if [ -f ../$outdir/kernel/epoch-${ne}.png ]; then
        continue
    fi

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
        ./train.py -i $image -o $outdir -k $kernel -kl 21 -sw ${sw} \
        -isz 4 -bs ${bs} -e ${ne} -w 0 -lr ${lr} -lrdk ${lrdk[@]} \
        -lrdc ${lrdc[@]} -wd ${wd} -ps ${ps} -ns 0 -ie ${ie} \
        -knc ${knc} -knh ${knh} -knk ${knk}
done | rush -j 6 {}
