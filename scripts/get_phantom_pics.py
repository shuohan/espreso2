#!/usr/bin/env python

import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from image_processing_3d import quantile_scale
from scipy.ndimage import zoom


dirname = '/data/phantom/data'
basename = 'SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-4mm-gapn2mm_resampled.nii'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

th_slice_ind = 130
in_slice_ind = 30 

factor = 2

ysize = 119
xsize = 60
xstart = 60
ystart = 40

in_im = data[:, :, in_slice_ind].astype(np.uint8).T
in_im = in_im[xstart : xstart + xsize, ystart : ystart + ysize]

xstart = 90
ystart = 40
th_im = data[th_slice_ind, :, :].astype(np.uint8).T
th_im = zoom(th_im, (factor, 1), order=0, prefilter=False)
th_im = th_im[xstart : xstart + xsize, ystart : ystart + ysize]

in_im = Image.fromarray(in_im)
in_im.save('phantom_4_2_in.png')
th_im = Image.fromarray(th_im)
th_im.save('phantom_4_2_th.png')

basename = 'SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-4mm-gap1mm_resampled.nii'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

factor = 5
in_slice_ind = 12

xstart = 60
ystart = 40
in_im = data[:, :, in_slice_ind].astype(np.uint8).T
in_im = in_im[xstart : xstart + xsize, ystart : ystart + ysize]

xstart = 90
ystart = 40
th_im = data[th_slice_ind, :, :].astype(np.uint8).T
th_im = zoom(th_im, (factor, 1), order=0, prefilter=False)
th_im = th_im[xstart : xstart + xsize, ystart : ystart + ysize]

in_im = Image.fromarray(in_im)
in_im.save('phantom_4_5_in.png')
th_im = Image.fromarray(th_im)
th_im.save('phantom_4_5_th.png')
