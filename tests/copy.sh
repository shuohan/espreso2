#!/usr/bin/env bash

sourcedir=results_try-neg_sm-init
targetdir=~/Dropbox/presentations/slice_profile/results

for profile in $sourcedir/*/kernel/avg_epoch-10000.png; do
    fwhm=$(echo $profile | sed "s/.*\(fwhm-.*\)_scale.*/\1/")
    scale=$(echo $profile | sed "s/.*\(scale-.*\)_len.*/\1/")
    target_filename=${fwhm}_${scale}_$(basename $profile)
    cp $profile $targetdir/$target_filename
done
