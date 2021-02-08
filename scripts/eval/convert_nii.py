#!/usr/bin/env python

import re
import numpy as np
import nibabel as nib
from scipy.io import savemat
import os


filenames = ['/data/spest/ismore/sub-OAS30016_ses-d0021_acq-mprage_T1w_type-gauss_fwhm-3p2_scale-0p25_len-13_ismore/output_image.nii',
             '/data/spest/ismore_sp/sub-OAS30016_ses-d0021_acq-mprage_T1w_type-gauss_fwhm-3p2_scale-0p25_len-13_ismore/output_image.nii',
             '/data/oasis3/data/sub-OAS30016_ses-d0021_acq-mprage_T1w.nii.gz']

for fn in filenames:
    array = nib.load(fn).get_fdata(dtype=np.float32)
    output_fn = os.path.basename(fn).replace('.nii', '.mat')
    output_fn = fn.replace('/output_image.nii', '')
    output_fn = output_fn.replace('.nii.gz', '')
    if 'ismore' in output_fn:
        name = re.sub(r'.*fwhm-([0-9p]*)_.*', r'\1', os.path.basename(output_fn))
        if '_sp' in output_fn:
            proc_type = 'sp'
        else:
            proc_type = ''
        name = 'image_' + name + proc_type
        output_fn = os.path.join('images', os.path.basename(output_fn))
        output_fn = output_fn + proc_type + '.mat'
    else:
        name = 'image_ref'
        output_fn = os.path.join('images', os.path.basename(output_fn)) + '.mat'
    print(name, output_fn)
    savemat(output_fn, {name: array})
