#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib 

from spest.utils import calc_fwhm


def plot(est, filename):

    font = {'size': 8}
    matplotlib.rc('font', **font)

    est_hm = np.max(est) / 2
    est_fwhm, est_left, est_right = calc_fwhm(est)

    dpi = 100
    figx = 90
    figy = 90

    figl = 0.30
    figr = 0.05
    figb = 0.25
    figt = 0.05
    position = [figl, figb, 1 - figl - figr, 1 - figb - figt]

    fig = plt.figure(figsize=(figx/dpi, figy/dpi), dpi=dpi)
    ax = fig.add_subplot(111, position=position)

    plt.plot(est, '-', color='tab:blue')
    plt.plot([est_left, est_right], [est_hm] * 2, '--o', color='tab:blue',
             markersize=3)

    ylim = ax.get_ylim()
    print(np.diff(ylim)[0] / ((1 - figt - figb) * figy))
    offset = np.diff(ylim)[0] / ((1 - figt - figb) * figy) * 12
    print(offset)
    tl = est_right + 1.5#
    est_tv = est_hm

    plt.text(tl, est_tv, '%.2f' % est_fwhm, color='tab:blue',
             va='center')

    ylim = plt.gca().get_ylim()
    yticks = np.arange(0, ylim[-1], 0.1)
    plt.yticks(yticks)
    plt.xticks(np.arange(0, len(est), 5))
    # plt.yticks([0, 0.05, 0.10])
    # plt.ylim([-0.01, 0.15])

    plt.savefig(filename)


if __name__ == '__main__':
    
    est_filename = '/data/spest/ipmi_esitmate_smore_resolution/sub-OAS30016_ses-d0021_acq-mprage_T1w/type-gauss_fwhm-2p0_scale-0p5_len-13/kernel/avg_epoch-15000.npy'
    est = np.load(est_filename).squeeze()
    plt.plot(est)
    
    Path('measure').mkdir(exist_ok=True)
    plot(est, 'measure/sub-OAS30016_type-gauss_fwhm-2p0_scale-0p5_kernel.pdf')

    est_filename = '/data/spest/ipmi_esitmate_smore_resolution/sub-OAS30016_ses-d0021_acq-mprage_T1w/type-gauss_fwhm-4p0_scale-0p25_len-13/kernel/avg_epoch-15000.npy'
    est = np.load(est_filename).squeeze()
    plt.plot(est)
    
    Path('measure').mkdir(exist_ok=True)
    plot(est, 'measure/sub-OAS30016_type-gauss_fwhm-4p0_scale-0p25_kernel.pdf')
