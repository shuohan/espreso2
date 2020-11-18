#!/usr/bin/env bash

dir=../../figures
mkdir -p $dir

fwhm=(2p0 4p0 8p0)
scale=(0p25 0p5)

# for f in ${fwhm[@]}; do
#     for s in ${scale[@]}; do
#     ./plot_curves.py -i \
#         ../../results/simu_lr-2e-4_bs-32_lrdk-3-1_ne-5000_ns-1000_sw-1/type-gauss_fwhm-${f}_scale-${s}_len-13/evseg_segal_loss.csv \
#         ../../results/simu_lr-2e-4_bs-32_lrdk-3-1_ne-5000_ns-5000_sw-1/type-gauss_fwhm-${f}_scale-${s}_len-13/evseg_segal_loss.csv \
#         ../../results/simu_lr-2e-4_bs-32_lrdk-3-1_ne-5000_ns-1000_sw-0/type-gauss_fwhm-${f}_scale-${s}_len-13/evseg_segal_loss.csv \
#         -x epoch -y mae -n \
#         ne-5000_ns-1000_sw-1 \
#         ne-5000_ns-5000_sw-1 \
#         ne-5000_ns-1000_sw-0 \
#         -m 1000 2000 3000 4000 5000 \
#         -o $dir/ne-5000_ns-1000_fwhm-${f}_scale-${s}.png
#     done
# done

for f in ${fwhm[@]}; do
    for s in ${scale[@]}; do
    ./plot_curves.py -i \
        ../../results/simu_lr-2e-4_bs-32_lrdk-3-1_ne-9000_ns-9000_sw-1/type-gauss_fwhm-${f}_scale-${s}_len-13/evseg_segal_loss.csv \
        ../../results/simu_lr-2e-4_bs-32_lrdk-3-1_ne-9000_ns-3000_sw-1/type-gauss_fwhm-${f}_scale-${s}_len-13/evseg_segal_loss.csv \
        ../../results/simu_lr-2e-4_bs-32_lrdk-3-1_ne-10000_ns-2000_sw-1/type-gauss_fwhm-${f}_scale-${s}_len-13/evseg_segal_loss.csv \
        -x epoch -y mae -n \
        ne-9000_ns-9000_sw-1 \
        ne-9000_ns-3000_sw-1 \
        ne-10000_ns-2000_sw-1 \
        -m 3000 6000 9000 \
        -m 2000 4000 6000 8000 10000 \
        -o $dir/ne-9000_ns-3000_fwhm-${f}_scale-${s}.png \
        -yl 0 0.02
    done
done
