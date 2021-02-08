#!/usr/bin/env python 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib 

from psf_est.utils import calc_fwhm


def plot(est, filename):

    font = {'size': 8}
    matplotlib.rc('font', **font)

    est_hm = np.max(est) / 2
    est_fwhm, est_left, est_right = calc_fwhm(est)

    dpi = 100
    figx = 168
    figy = 120

    figl = 0.24
    figr = 0.01
    figb = 0.20
    figt = 0.05
    position = [figl, figb, 1 - figl - figr, 1 - figb - figt]

    fig = plt.figure(figsize=(figx/dpi, figy/dpi), dpi=dpi)
    ax = fig.add_subplot(111, position=position)

    plt.plot(est, '-', color='tab:blue')
    plt.plot([est_left, est_right], [est_hm] * 2, '--o', color='tab:blue',
             markersize=5)

    tl = est_right + 1.5
    plt.text(tl, est_hm, '%.2f' % est_fwhm, color='tab:blue',
             va='center')

    plt.xticks(np.arange(0, len(est), 4))
    plt.yticks([0, 0.1, 0.2])
    plt.ylim([-0.015, 0.3])

    plt.savefig(filename)


if __name__ == '__main__':
    
    est_dirname =  '../tests/results_isbi2021_phantom'
    est_basename = 'phantom_4mm_gapn2mm_smooth-1.0/kernel/avg_epoch-20000.npy'
    est_filename = Path(est_dirname, est_basename)
    est = np.load(est_filename).squeeze()

    plot(est, 'phantom_4_2_kernel.pdf')

    est_basename = 'phantom_4mm_gap1mm_smooth-1.0/kernel/avg_epoch-20000.npy'
    est_filename = Path(est_dirname, est_basename)
    est = np.load(est_filename).squeeze()

    plot(est, 'phantom_4_5_kernel.pdf')
