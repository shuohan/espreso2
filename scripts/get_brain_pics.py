#!/usr/bin/env python

import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from image_processing_3d import quantile_scale
from scipy.ndimage import zoom


filename = '20121_02_FLAIRPre_2D.nii'
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
zooms = obj.header.get_zooms()
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

print(data.shape)

th_slice_ind = 135
in_slice_ind = 40

factor = zooms[2] / zooms[0]

ysize = 119
xsize = 90
# ysize = 200
# xsize = 200
xstart = 25
ystart = 40

in_im = data[:, :, in_slice_ind].astype(np.uint8).T
in_im = in_im[xstart : xstart + xsize, ystart : ystart + ysize]

xstart = 10
ystart = 10
th_im = data[th_slice_ind, :, :].astype(np.uint8).T
th_im = th_im[::-1, :]
th_im = zoom(th_im, (factor, 1), order=0, prefilter=False)
th_im = th_im[xstart : xstart + xsize, ystart : ystart + ysize]

in_im = Image.fromarray(in_im)
in_im.save('flair_in.png')
th_im = Image.fromarray(th_im)
th_im.save('flair_th.png')

filename = '20208_03_T1Post.nii'
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255
zooms = obj.header.get_zooms()

factor = zooms[2] / zooms[0]

in_slice_ind = 30

xstart = 35
ystart = 40
in_im = data[:, :, in_slice_ind].astype(np.uint8).T
in_im = in_im[xstart : xstart + xsize, ystart : ystart + ysize]

xstart = 30
ystart = 20
th_im = data[th_slice_ind, :, :].astype(np.uint8).T
th_im = th_im[::-1, :]
th_im = zoom(th_im, (factor, 1), order=0, prefilter=False)
th_im = th_im[xstart : xstart + xsize, ystart : ystart + ysize]

in_im = Image.fromarray(in_im)
in_im.save('t1_in.png')
th_im = Image.fromarray(th_im)
th_im.save('t1_th.png')
