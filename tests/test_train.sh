#!/usr/bin/env bash

espreso2_dir=$(realpath $(dirname $0)/..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
improc3d_dir=~/Code/shuo/utils/improc3d
ptxl_dir=~/Code/shuo/deep-networks/ptxl
data_dir=/data

export PYTHONPATH=${espreso2_dir}:${sssrlib_dir}:${improc3d_dir}:${ptxl_dir}
export CUDA_VISIBLE_DEVICES=1

# images=(/data/smore_simu_same_fov/simu_data/scale-2p0_fwhm-2p0/sub-OAS30004_ses-d1101_T2w_initnorm_scale-2p0_fwhm-2p0.nii.gz)
# images=(/data/smore_simu_same_fov/simu_data/scale-2p0_fwhm-1p0/sub-OAS30004_ses-d1101_T2w_initnorm_scale-2p0_fwhm-1p0.nii.gz)
# images=(/data/smore_simu_same_fov/orig_data/sub-OAS30004_ses-d1101_T2w_initnorm.nii.gz)
# images=(/data/smore_simu_same_fov/espreso2_sample_valid/sub-OAS30050_ses-d0110_T2w_initnorm_scale-4p9_fwhm-2p45.nii.gz
#         /data/smore_simu_same_fov/espreso2_sample_valid/sub-OAS30050_ses-d0110_T2w_initnorm_scale-4p9_fwhm-6p125.nii.gz
#         /data/smore_simu_same_fov/espreso2_sample_valid/sub-OAS30050_ses-d0110_T2w_initnorm_scale-4p9_fwhm-7.nii.gz)
# images=(/data/smore_simu_same_fov/espreso2_sample_valid/sub-OAS30050_ses-d0110_T2w_initnorm_scale-4p9_fwhm-7.nii.gz)
images=($(ls -r /data/smore_simu_same_fov/espreso2_sample_valid/sub-OAS30050_ses-d0110_T2w_initnorm_*.nii.gz))

ni=2000
nu=80
ul=1e-4

for image in ${images[@]}; do
    fwhm=$(echo $image | sed "s/.*\(fwhm-[0-9p]*\).*/\1/")
    scale=$(echo $image | sed "s/.*\(scale-[0-9p]*\).*/\1/")
    outdir=results_train/nu-${nu}_ul-${ul}/$(basename $image | sed "s/\.nii\.gz//")
    sp=$(echo $image | sed "s/\.nii\.gz/.npy/")
    # espreso2-train -i $image -o $outdir -I ${ni} -p $sp -Z 4 -P 16 -g \
    #     -s 1000 -M foreground
    set -x
    ../scripts/train.py -i $image -o $outdir -Z 4 \
        -s 500 -p ${sp} -g -e 100 -u $nu -m -N ${ul} -T 0
    set +x
done

# docker run --gpus device=1 --rm \
#         -v $data_dir:$data_dir \
#         -v $psf_est_dir:$psf_est_dir \
#         -v $sssrlib_dir:$sssrlib_dir \
#         -v $proc_dir:$proc_dir \
#         -v $trainer_dir:$trainer_dir \
#         -v $config_dir:$config_dir \
#         --user $(id -u):$(id -g) \
#         -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir \
#         -w $psf_est_dir/scripts -t \
#         pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime \
