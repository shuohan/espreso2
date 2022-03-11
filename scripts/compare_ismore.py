#!/usr/bin/env python

import re
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

from ssim_and_psnr_3d import calc_mask, calc_psnr_3d, calc_ssim_3d


ref_dir = Path('/data/oasis3/data')
with_sp_dir = Path('/data/spest/ismore_with_sp')
without_sp_dir = Path('/data/spest/ismore_without_sp')

output_dir = Path('compare_with_and_without_sp')
output_dir.mkdir(exist_ok=True)


df = list()


for with_image_fn in sorted(with_sp_dir.iterdir()):
    without_image_fn = with_image_fn.stem.replace('_ismore_sp', '')
    without_image_fn = without_sp_dir.joinpath(without_image_fn)

    prefix = without_image_fn.stem
    fwhm = re.sub(r'.*fwhm-([0-9p]*).*', r'\1', without_image_fn.stem)
    fwhm = float(fwhm.replace('p', '.'))
    ref_image_prefix = re.sub(r'_type.*$', '', prefix)
    ref_image_fn = Path(ref_dir, ref_image_prefix).with_suffix('.nii.gz')

    without_image_fn = without_image_fn.joinpath('output_image.nii')
    with_image_fn = with_image_fn.joinpath('output_image.nii')

    # print(with_image_fn, without_image_fn, ref_image_fn)

    ref_image = nib.load(ref_image_fn).get_fdata(dtype=np.float32)
    with_image = nib.load(with_image_fn).get_fdata(dtype=np.float32)
    without_image = nib.load(without_image_fn).get_fdata(dtype=np.float32)

    mask = calc_mask(ref_image)

    fig = plt.figure(figsize=(10, 10))

    plt.subplot(4, 3, 1)
    plt.imshow(mask[:, :, mask.shape[2]//2], cmap='gray')
    plt.title('mask')
    plt.subplot(4, 3, 2)
    plt.imshow(mask[:, mask.shape[1]//2, :], cmap='gray')
    plt.subplot(4, 3, 3)
    plt.imshow(mask[mask.shape[0]//2, :, :], cmap='gray')

    plt.subplot(4, 3, 4)
    plt.title('ref image')
    plt.imshow(ref_image[:, :, ref_image.shape[2]//2], cmap='gray')
    plt.subplot(4, 3, 5)
    plt.imshow(ref_image[:, ref_image.shape[1]//2, :], cmap='gray')
    plt.subplot(4, 3, 6)
    plt.imshow(ref_image[ref_image.shape[0]//2, :, :], cmap='gray')

    plt.subplot(4, 3, 7)
    plt.title('without image')
    plt.imshow(without_image[:, :, without_image.shape[2]//2], cmap='gray')
    plt.subplot(4, 3, 8)
    plt.imshow(without_image[:, without_image.shape[1]//2, :], cmap='gray')
    plt.subplot(4, 3, 9)
    plt.imshow(without_image[without_image.shape[0]//2, :, :], cmap='gray')

    plt.subplot(4, 3, 10)
    plt.title('with image')
    plt.imshow(with_image[:, :, with_image.shape[2]//2], cmap='gray')
    plt.subplot(4, 3, 11)
    plt.imshow(with_image[:, with_image.shape[1]//2, :], cmap='gray')
    plt.subplot(4, 3, 12)
    plt.imshow(with_image[with_image.shape[0]//2, :, :], cmap='gray')

    figure_fn = Path(output_dir, prefix).with_suffix('.png')
    fig.savefig(figure_fn)

    without_psnr = calc_psnr_3d(ref_image, without_image, mask)
    with_psnr = calc_psnr_3d(ref_image, with_image, mask)
    without_ssim = calc_ssim_3d(ref_image, without_image, mask)
    with_ssim = calc_ssim_3d(ref_image, with_image, mask)

    print(prefix, fwhm)
    print('psnr without %g, with %g' % (without_psnr, with_psnr))
    print('ssim without %g, with %g' % (without_ssim, with_ssim))

    tab = OrderedDict([('Subject', ref_image_prefix),
                       ('FWHM', fwhm),
                       ('W/ SP', False),
                       ('PSNR', without_psnr),
                       ('SSIM', without_ssim)])
    df.append(tab)

    tab = OrderedDict([('Subject', ref_image_prefix),
                       ('FWHM', fwhm),
                       ('W/ SP', True),
                       ('PSNR', with_psnr),
                       ('SSIM', with_ssim)])
    df.append(tab)


df = pd.DataFrame(df)
print(df)
df.to_csv(Path(output_dir, 'dataframe.csv'), index=False)
