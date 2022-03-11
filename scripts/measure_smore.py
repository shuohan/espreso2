#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

from spest.utils import calc_fwhm


subjects = ['sub-OAS30016_ses-d0021_acq-mprage_T1w',
            'sub-OAS30032_ses-d3499_acq-mprage_T1w',
            'sub-OAS30048_ses-d3367_acq-mprage_T1w',
            'sub-OAS30064_ses-d0687_acq-mprage_run-02_T1w',
            'sub-OAS30080_ses-d0048_acq-mprage_T1w']

top_est_dirname = '/data/spest/ipmi_esitmate_smore_resolution'
output_dirname = Path('measure')
output_dirname.mkdir(exist_ok=True)

fwhms = ['2p0', '4p0']
scales = ['0p5', '0p25']

df = list()

for subj in subjects:
    for f, s in zip(fwhms, scales):
        est_dirname = 'type-gauss_fwhm-{}_scale-{}_len-13'.format(f, s)
        est_filename = Path(top_est_dirname, subj, est_dirname, 'kernel', 'avg_epoch-15000.npy')
        est_kernel = np.load(est_filename)
        est_fwhm = calc_fwhm(est_kernel)[0]

        tab = OrderedDict([('subject', subj),
                           ('simu fwhm', f.replace('p', '.')),
                           ('simu scale', s.replace('p', '.')),
                           ('measure fwhm', est_fwhm)])
        df.append(tab)

df = pd.DataFrame(df)
print(df)
df.to_csv(Path(output_dirname, 'dataframe.csv'), index=False)

df_mean = df.groupby(['simu fwhm', 'simu scale']).mean()
print(df_mean)

df_std = df.groupby(['simu fwhm', 'simu scale']).std()
print(df_std)
