#!/usr/bin/env bash

dir=../../figures
mkdir -p $dir

./plot_curves.py -i \
    ../../tests/simu_fixed-1/simu_fwhm-4p0_scale-0p25_len-13/evseg_segal_loss.csv \
    ../../tests/rot-0-0-0/simu_fixed-1/simu_fwhm-4p0_scale-0p25_len-13/evseg_segal_loss.csv \
    ../../tests/simu_fixed-4/simu_fwhm-4p0_scale-0p25_len-13/evseg_segal_loss.csv \
    ../../tests/rot-0-0-0/simu_fixed-0p25/simu_fwhm-4p0_scale-0p25_len-13/evseg_segal_loss.csv \
    -x epoch -y mae -n \
    rot-0-45-45_fix-1 \
    rot-0-0-0_fix-1 \
    rot-0-45-45_fix-4 \
    rot-0-0-0_fix-4 \
    -o $dir/rot.png \
    -yl 0.0025 0.0150
