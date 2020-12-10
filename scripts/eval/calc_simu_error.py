#!/usr/bin/env python

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

import matplotlib.pyplot as plt
from collections import OrderedDict

from spest.utils import calc_fwhm
from lr_simu.simu import ThroughPlaneSimulatorCPU
from ssim_and_psnr_3d import calc_mask, calc_psnr_3d, calc_ssim_3d

import sys


if len(sys.argv) == 1:
    num_epochs = 10000
else:
    num_epochs = int(sys.argv[1])

# subjects = ['sub-OAS30016_ses-d0021_acq-mprage_T1w']
subjects = ['sub-OAS30016_ses-d0021_acq-mprage_T1w',
            'sub-OAS30032_ses-d3499_acq-mprage_T1w',
            'sub-OAS30048_ses-d3367_acq-mprage_T1w',
            'sub-OAS30064_ses-d0687_acq-mprage_run-02_T1w',
            'sub-OAS30080_ses-d0048_acq-mprage_T1w']

top_est_dirname = '/data/spest/ipmi_simu_test'
true_dirname = '/data/oasis3/data'
simu_dirname = '/data/oasis3/simu'
output_dirname = 'errors_epoch-{}'.format(num_epochs)

types = ['gauss', 'rect']
fwhms = {'gauss': ['2p0', '4p0', '8p0'], 'rect': ['3p0', '5p0', '9p0']}
scales = ['0p5', '0p25', '0p125']

df = list()

for subj in subjects:

    true_filename = Path(true_dirname, subj).with_suffix('.nii.gz')
    iso_image = nib.load(true_filename).get_fdata(dtype=np.float32)

    max_val = np.max(iso_image)
    min_val = np.min(iso_image)

    assert true_filename.is_file()
    est_basename = 'avg_epoch-{}.npy'.format(num_epochs)
    for t in types:
        for f in fwhms[t]:
            for s in scales:
                est_dirname = 'type-{}_fwhm-{}_scale-{}_len-13'.format(t, f, s)
                est_filename = Path(top_est_dirname, subj, est_dirname, 'kernel', est_basename)
                simu_filename = '{}_type-{}_fwhm-{}_scale-{}_len-13_kernel.npy'.format(subj, t, f, s)
                simu_filename = Path(simu_dirname, simu_filename)

                est_kernel = np.load(est_filename)
                ref_kernel = np.load(simu_filename)

                left = (len(est_kernel) - len(ref_kernel)) // 2
                right = len(est_kernel) - len(ref_kernel) - left
                ref_kernel = np.pad(ref_kernel, (left, right))

                est_fwhm = calc_fwhm(est_kernel)[0]
                ref_fwhm = calc_fwhm(ref_kernel)[0]

                scale = float(s.replace('p', '.'))
                est_simulator = ThroughPlaneSimulatorCPU(est_kernel, scale_factor=scale)
                ref_simulator = ThroughPlaneSimulatorCPU(ref_kernel, scale_factor=scale)

                est_image = est_simulator.simulate(iso_image)
                ref_image = ref_simulator.simulate(iso_image)

                mask = calc_mask(ref_image)

                Path(output_dirname, subj).mkdir(exist_ok=True, parents=True)

                fig = plt.figure()
                plt.subplot(1, 3, 1)
                plt.imshow(mask[:, :, mask.shape[2]//2], cmap='gray')
                plt.subplot(1, 3, 2)
                plt.imshow(mask[:, mask.shape[1]//2, :], cmap='gray')
                plt.subplot(1, 3, 3)
                plt.imshow(mask[mask.shape[0]//2, :, :], cmap='gray')

                filename = '{}_{}_{}_mask.png'.format(t, s, f)
                filename = Path(output_dirname, subj, filename)
                fig.savefig(filename)
                print(filename)

                fwhm_error = np.abs(est_fwhm - ref_fwhm)
                prof_error = np.sum(np.abs(est_kernel - ref_kernel))

                psnr = calc_psnr_3d(ref_image, est_image, mask, [min_val, max_val])
                ssim = calc_ssim_3d(ref_image, est_image, mask)

                fig = plt.figure()

                plt.subplot(2, 3, 1)
                plt.imshow(est_image[:, :, est_image.shape[2]//2], cmap='gray')
                plt.subplot(2, 3, 2)
                plt.imshow(est_image[:, est_image.shape[1]//2, :], cmap='gray')
                plt.subplot(2, 3, 3)
                plt.imshow(est_image[est_image.shape[0]//2, :, :], cmap='gray')
                      
                plt.subplot(2, 3, 4)
                plt.imshow(ref_image[:, :, ref_image.shape[2]//2], cmap='gray')
                plt.subplot(2, 3, 5)
                plt.imshow(ref_image[:, ref_image.shape[1]//2, :], cmap='gray')
                plt.subplot(2, 3, 6)
                plt.imshow(ref_image[ref_image.shape[0]//2, :, :], cmap='gray')

                filename = '{}_{}_{}_error.png'.format(t, s, f)
                filename = Path(output_dirname, subj, filename)
                fig.savefig(filename)
                print(filename)

                tab = OrderedDict([('subject', subj),
                                   ('type', t),
                                   ('fwhm', f.replace('p', '.')),
                                   ('scale', s.replace('p', '.')),
                                   ('fwhm error', fwhm_error),
                                   ('profile error', prof_error),
                                   ('psnr', psnr),
                                   ('ssim', ssim)])

                df.append(tab)

df = pd.DataFrame(df)
print(df)

df.to_csv(Path(output_dirname, 'dataframe.csv'), index=False)
