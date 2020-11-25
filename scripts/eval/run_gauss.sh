#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/../..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
simu_dir=~/Code/shuo/utils/lr-simu
data_dir=/data

images=(/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_rot-0-45-45_type-gauss_fwhm-8p0_scale-0p125_len-13.nii
        /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_rot-0-45-45_type-gauss_fwhm-4p0_scale-0p125_len-13.nii
        /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_rot-0-45-45_type-gauss_fwhm-2p0_scale-0p125_len-13.nii)
# images=(/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_rot-0-45-45_type-gauss_fwhm-8p0_scale-0p5_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_rot-0-45-45_type-gauss_fwhm-4p0_scale-0p5_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_rot-0-45-45_type-gauss_fwhm-2p0_scale-0p5_len-13.nii)
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-2p0_scale-0p125_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-4p0_scale-0p125_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-8p0_scale-0p125_len-13.nii

lr=2e-4
bs=32
ne=10000
lrdk=(3 1)
sw=0
sw_str=$(echo $sw | sed "s/\./p/")

for image in ${images[@]}; do
    fwhm=$(echo $image | sed "s/.*\(fwhm-.*\)_scale.*/\1/")
    scale=$(echo $image | sed "s/.*\(scale-.*\)_len.*/\1/")
    kernel=$(echo $image | sed "s/.*\(type-.*\)_fw.*/\1/")
    len=$(echo $image | sed "s/.*\(len-.*\)\.nii/\1/")
    outdir=../results/simu_lr-${lr}_bs-${bs}_lrdk-${lrdk[0]}-${lrdk[1]}_ne-${ne}_sw-${sw_str}_transpose_single-side-loss/${kernel}_${fwhm}_${scale}_${len}
    if [ -f $outdir/kernel/epoch-${ne}.png ]; then
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
        ./train.py -i $image -o $outdir -k $kernel -kl 19 -sw ${sw} \
        -isz 4 -bs ${bs} -e ${ne} -w 0 -lr ${lr} -lrdk ${lrdk[0]} ${lrdk[1]}
done | rush -j 3 {}
