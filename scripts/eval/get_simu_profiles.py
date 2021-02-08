#!/usr/bin/env python 

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib 

from spest.utils import calc_fwhm


def plot(est, ref, filename):

    left = (len(est) - len(ref)) // 2
    right = len(est) - len(ref) - left
    ref = np.pad(ref, (left, right))

    font = {'size': 8}
    matplotlib.rc('font', **font)

    est_hm = np.max(est) / 2
    est_fwhm, est_left, est_right = calc_fwhm(est)

    ref_hm = np.max(ref) / 2
    ref_fwhm, ref_left, ref_right = calc_fwhm(ref)

    dpi = 100
    figx = 143
    figy = 90

    figl = 0.25
    figr = 0.01
    figb = 0.21
    figt = 0.05
    position = [figl, figb, 1 - figl - figr, 1 - figb - figt]

    fig = plt.figure(figsize=(figx/dpi, figy/dpi), dpi=dpi)
    ax = fig.add_subplot(111, position=position)

    plt.plot(ref, '-', color='tab:red')
    plt.plot([ref_left, ref_right], [ref_hm] * 2, '--o', color='tab:red',
             markersize=5)

    plt.plot(est, '-', color='tab:blue')
    plt.plot([est_left, est_right], [est_hm] * 2, '--o', color='tab:blue',
             markersize=5)

    ylim = ax.get_ylim()
    print(np.diff(ylim)[0] / ((1 - figt - figb) * figy))
    offset = np.diff(ylim)[0] / ((1 - figt - figb) * figy) * 12
    print(offset)
    tl = np.max((est_right, ref_right)) + 1.5
    est_tv = (est_hm + ref_hm) * 0.5 + offset / 2
    ref_tv = (est_hm + ref_hm) * 0.5 - offset / 2

    plt.text(tl, ref_tv, '%.2f' % ref_fwhm, color='tab:red',
             va='center')
    plt.text(tl, est_tv, '%.2f' % est_fwhm, color='tab:blue',
             va='center')

    ylim = plt.gca().get_ylim()
    q1 = (ylim[1] - ylim[0]) / 3
    q2 = q1 * 2
    yticks = np.round([0, q1, q2], 2)
    plt.yticks(yticks)
    plt.xticks(np.arange(0, len(est), 4))
    # plt.yticks([0, 0.05, 0.10])
    # plt.ylim([-0.01, 0.15])

    plt.savefig(filename)


if __name__ == '__main__':
    
    est_filename = '/data/spest/ipmi_simu_test/sub-OAS30016_ses-d0021_acq-mprage_T1w/type-gauss_fwhm-2p0_scale-0p25_len-13/kernel/avg_epoch-15000.npy'
    est = np.load(est_filename).squeeze()

    ref_filename = '/data/oasis3/simu/sub-OAS30016_ses-d0021_acq-mprage_T1w_type-gauss_fwhm-2p0_scale-0p25_len-13_kernel.npy'
    ref = np.load(ref_filename).squeeze()
    
    Path('simu_figures').mkdir(exist_ok=True)
    plot(est, ref, 'simu_figures/sub-OAS30016_type-gauss_fwhm-2p0_scale-0p25_kernel.pdf')

    # --------------
    est_filename = '/data/spest/ipmi_simu_test/sub-OAS30032_ses-d3499_acq-mprage_T1w/type-rect_fwhm-5p0_scale-0p25_len-13/kernel/avg_epoch-15000.npy'
    est = np.load(est_filename).squeeze()

    ref_filename = '/data/oasis3/simu/sub-OAS30032_ses-d3499_acq-mprage_T1w_type-rect_fwhm-5p0_scale-0p25_len-13_kernel.npy'
    ref = np.load(ref_filename).squeeze()
    
    Path('simu_figures').mkdir(exist_ok=True)
    plot(est, ref, 'simu_figures/sub-OAS30032_type-rect_fwhm-5p0_scale-0p25_kernel.pdf')

    # --------------
    est_filename = '/data/spest/ipmi_simu_test/sub-OAS30048_ses-d3367_acq-mprage_T1w/type-gauss_fwhm-4p0_scale-0p5_len-13/kernel/avg_epoch-15000.npy'
    est = np.load(est_filename).squeeze()

    ref_filename = '/data/oasis3/simu/sub-OAS30048_ses-d3367_acq-mprage_T1w_type-gauss_fwhm-4p0_scale-0p5_len-13_kernel.npy'
    ref = np.load(ref_filename).squeeze()
    
    Path('simu_figures').mkdir(exist_ok=True)
    plot(est, ref, 'simu_figures/sub-OAS30048_type-gauss_fwhm-4p0_scale-0p5_kernel.pdf')

    # --------------
    est_filename = '/data/spest/ipmi_simu_test/sub-OAS30080_ses-d0048_acq-mprage_T1w/type-rect_fwhm-9p0_scale-0p125_len-13/kernel/avg_epoch-15000.npy'
    est = np.load(est_filename).squeeze()

    ref_filename = '/data/oasis3/simu/sub-OAS30080_ses-d0048_acq-mprage_T1w_type-rect_fwhm-9p0_scale-0p125_len-13_kernel.npy'
    ref = np.load(ref_filename).squeeze()
    
    Path('simu_figures').mkdir(exist_ok=True)
    plot(est, ref, 'simu_figures/sub-OAS30080_type-gauss_fwhm-9p0_scale-0p125_kernel.pdf')
