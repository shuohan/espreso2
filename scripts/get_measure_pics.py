#!/usr/bin/env python

import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from image_processing_3d import quantile_scale
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


dirname = '/data/spest/ismore_without_sp/'
basename = 'sub-OAS30016_ses-d0021_acq-mprage_T1w_type-gauss_fwhm-2p0_scale-0p5_len-13'
filename = Path(dirname, basename, 'output_image.nii')
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

hr_ps = 64
lr_ps = 68
xstart = 40
zstart = 150

y = 130

im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
im = im[::-1, :]
print(im.shape)

# plt.imshow(im, cmap='gray')
# plt.show()

im = Image.fromarray(im)
im.save('measure/sub-OAS30016_2_ismore.png')

# ----

dirname = '/data/spest/ismore_without_sp/'
basename = 'sub-OAS30016_ses-d0021_acq-mprage_T1w_type-gauss_fwhm-4p0_scale-0p25_len-13'
filename = Path(dirname, basename, 'output_image.nii')
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
im = im[::-1, :]
print(im.shape)

# plt.imshow(im, cmap='gray')
# plt.show()

im = Image.fromarray(im)
im.save('measure/sub-OAS30016_4_ismore.png')

# ----
dirname = '/data/oasis3/data'
basename = 'sub-OAS30016_ses-d0021_acq-mprage_T1w.nii.gz'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
im = im[::-1, :]
print(im.shape)

im = Image.fromarray(im)
im.save('measure/sub-OAS30016_truth.png')

# 
# dirname = '/data/spest/ismore_without_sp/'
# basename = 'sub-OAS30032_ses-d3499_acq-mprage_T1w_type-gauss_fwhm-2p0_scale-0p5_len-13'
# filename = Path(dirname, basename, 'output_image.nii')
# obj = nib.load(filename)
# data = obj.get_fdata(dtype=np.float32)
# data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255
# 
# hr_ps = 64
# lr_ps = 68
# xstart = 40
# zstart = 150
# 
# y = 130
# 
# im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
# im = im[::-1, :]
# print(im.shape)
# 
# # plt.imshow(im, cmap='gray')
# # plt.show()
# 
# im = Image.fromarray(im)
# im.save('measure/sub-OAS30032_ismore.png')
# 
# ---- 

# dirname = '/data/oasis3/data'
# basename = 'sub-OAS30032_ses-d3499_acq-mprage_T1w.nii.gz'
# filename = Path(dirname, basename)
# obj = nib.load(filename)
# data = obj.get_fdata(dtype=np.float32)
# data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255
# 
# im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
# im = im[::-1, :]
# print(im.shape)
# 
# im = Image.fromarray(im)
# im.save('measure/sub-OAS30032_truth.png')
# 
# # ---
# 
# dirname = '/data/spest/ismore_without_sp'
# basename = 'sub-OAS30064_ses-d0687_acq-mprage_run-02_T1w_type-gauss_fwhm-4p0_scale-0p25_len-13'
# filename = Path(dirname, basename, 'output_image.nii')
# obj = nib.load(filename)
# data = obj.get_fdata(dtype=np.float32)
# data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255
# 
# hr_ps = 68
# lr_ps = 68
# xstart = 40
# zstart = 120
# 
# y = 130
# 
# im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
# im = im[::-1, :]
# print(im.shape)
# 
# im = Image.fromarray(im)
# im.save('measure/sub-OAS30064_ismore.png')
# 
# # ---
# 
# dirname = '/data/oasis3/data'
# basename = 'sub-OAS30064_ses-d0687_acq-mprage_run-02_T1w.nii.gz'
# filename = Path(dirname, basename)
# obj = nib.load(filename)
# data = obj.get_fdata(dtype=np.float32)
# data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255
# 
# im = data[xstart : xstart + hr_ps, y, zstart : zstart + lr_ps].astype(np.uint8).T
# im = im[::-1, :]
# print(im.shape)
# 
# im = Image.fromarray(im)
# im.save('measure/sub-OAS30064_truth.png')
