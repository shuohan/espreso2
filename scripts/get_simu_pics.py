#!/usr/bin/env python

import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from image_processing_3d import quantile_scale
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


dirname = '/data/oasis3/simu'
basename = 'sub-OAS30016_ses-d0021_acq-mprage_T1w_type-gauss_fwhm-2p0_scale-0p25_len-13.nii'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

factor = 4
hr_ps = 64
lr_ps = int(hr_ps / factor)
xstart = 10
zstart = 30

y = 130

im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
im = zoom(im, (factor, 1), order=0, prefilter=False)[::-1, :]

im = Image.fromarray(im)
im.save('simu_figures/sub-OAS30016_image.png')

# -----------

basename = 'sub-OAS30032_ses-d3499_acq-mprage_T1w_type-rect_fwhm-5p0_scale-0p25_len-13.nii'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

factor = 4
hr_ps = 64
lr_ps = int(hr_ps / factor)
xstart = 50
zstart = 30

y = 130

im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
im = zoom(im, (factor, 1), order=0, prefilter=False)[::-1, :]

im = Image.fromarray(im)
im.save('simu_figures/sub-OAS30032_image.png')


# ---------
basename = 'sub-OAS30080_ses-d0048_acq-mprage_T1w_type-rect_fwhm-9p0_scale-0p125_len-13.nii'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

factor = 8
hr_ps = 64
lr_ps = int(hr_ps / factor)
xstart = 30
zstart = 18

y = 128
print(data.shape)

im = data[y, xstart : xstart + hr_ps, zstart : zstart + lr_ps].astype(np.uint8).T
im = zoom(im, (factor, 1), order=0, prefilter=False)[::-1, :]

im = Image.fromarray(im)
im.save('simu_figures/sub-OAS30080_image.png')

# ---------

basename = 'sub-OAS30048_ses-d3367_acq-mprage_T1w_type-gauss_fwhm-4p0_scale-0p5_len-13.nii'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

factor = 2
hr_ps = 64
lr_ps = int(hr_ps / factor)
xstart = 100
zstart = 50

y = 128

im = data[y, xstart : xstart + hr_ps, zstart : zstart + lr_ps].astype(np.uint8).T
im = zoom(im, (factor, 1), order=0, prefilter=False)[::-1, :]

im = Image.fromarray(im)
im.save('simu_figures/sub-OAS30048_image.png')
