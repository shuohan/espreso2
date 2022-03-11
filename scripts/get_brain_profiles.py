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

    figl = 0.20
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

    plt.xticks(np.arange(0, len(est), 2))
    plt.yticks([0, 0.1, 0.2, 0.3])
    plt.ylim([-0.02, 0.41])

    plt.savefig(filename)

if __name__ == '__main__':
    
    est_filename = '20121_02_FLAIRPre_2D_kernel.npy'
    est = np.load(est_filename).squeeze()

    plot(est, '20121_02_FLAIRPre_2D_kernel.pdf')

    est_filename = '20208_03_T1Post_kernel.npy'
    est = np.load(est_filename).squeeze()

    plot(est, '20208_03_T1Post_kernel.pdf')
