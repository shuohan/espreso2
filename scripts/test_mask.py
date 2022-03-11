#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter


filename = '/data/spest/ismore/sub-OAS30016_ses-d0021_acq-mprage_T1w_type-gauss_fwhm-2p0_scale-0p25_len-13_ismore/output_image.nii'
image = nib.load(filename).get_fdata(dtype=np.float32)

image = image / np.max(image)

blur_image = gaussian_filter(image, 2)

threshold = np.quantile(blur_image, 0.5)
binary = blur_image > threshold

image = image * binary

plt.subplot(1, 3, 1)
plt.imshow(image[:, :, binary.shape[2] // 2], cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(image[:, binary.shape[1] // 2, :], cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(image[binary.shape[0] // 2, :, :], cmap='gray')

plt.show()
