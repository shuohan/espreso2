#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/improc3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/ptxl
data_dir=/data

images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p5_len-13.nii)
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p125_len-13.nii)
# images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p25_len-13.nii)
# images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-8p0_scale-0p5_len-13.nii)
# images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-5p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-5p0_scale-0p5_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-3p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-5p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-9p0_scale-0p125_len-13.nii)
# images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p125_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p25_len-13.nii
#         /data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-2p0_scale-0p5_len-13.nii)
# images=(/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-rect_fwhm-5p0_scale-0p5_len-13.nii)

ns=1
sw=1
ie=100
wd=5e-2
in=1000
bs=16
ne=5000

for image in ${images[@]}; do
    fwhm=$(echo $image | sed "s/.*\(fwhm-.*\)_scale.*/\1/")
    scale=$(echo $image | sed "s/.*\(scale-.*\)_len.*/\1/")
    len=$(echo $image | sed "s/.*\(len-.*\)\.nii/\1/")
    outdir=../tests/results/oasis3_${fwhm}_${scale}_${len}_ns-${ns}_flip_sw-${sw}_ie-${ie}_wd-${wd}_in-${in}_bs-${bs}
    kernel=$(echo $image | sed "s/\.nii/_kernel.npy/")
    docker run --gpus device=1 --rm \
        -v $data_dir:$data_dir \
        -v $psf_est_dir:$psf_est_dir \
        -v $sssrlib_dir:$sssrlib_dir \
        -v $proc_dir:$proc_dir \
        -v $trainer_dir:$trainer_dir \
        -v $config_dir:$config_dir \
        --user $(id -u):$(id -g) \
        -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir \
        -w $psf_est_dir/scripts -t \
        pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime \
        ./train.py -d -i $image -o $outdir -k $kernel -kl 13 -isz 1 -e ${ne} \
        -sw ${sw} -wd ${sw} -lrdk 3,1 3,1 3,1 1,1 1,1 \
        -lrdc 64 64 64 64 -knc 3 -ns ${ns} -ps 40 -ie ${ie} \
        -wd ${wd} -in ${in} -bs ${bs} -css 5000  # -c $outdir/checkpoint/epoch-10.pt
        # -lrdc 128 128 128 128 128 128 128 128 128

done # | rush -j 3 {}
