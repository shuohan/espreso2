#!/usr/bin/env python

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from collections import OrderedDict

from psf_est.utils import calc_fwhm


est_dirname =  '../tests/results_isbi2021_phantom_final'
est_pattern = 'phantom_%smm_%smm_smooth-1.0/kernel/avg_epoch-30000.npy'

fwhm = ['2', '4']
scale = {'2': ['gapn0p5', 'gap0', 'gap0p5', 'gap1', 'gap2'],
         '4': ['gapn2', 'gapn1', 'gap0', 'gap1', 'gap2',]}

df = list()

for f in fwhm:
    for s in scale[f]:
        est_filename = Path(est_dirname, est_pattern % (f, s))
        est = np.load(est_filename).squeeze()
        est_fwhm = calc_fwhm(est)[0]
        tab = OrderedDict([('thick', f),
                           ('scale', s),
                           ('fwhm ', '%.2f' % est_fwhm)])
        df.append(tab)

df = pd.DataFrame(df).T
print(df)
