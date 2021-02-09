#!/usr/bin/env python

from pathlib import Path
import matplotlib.pyplot as plt


for image in Path('/home/shuo/Dropbox/Manuscripts/2021-ipmi-shan-slice-profile/figures/flowchart').glob('*prob*'):
    im = plt.imread(image)
    output_basename = str(image.stem) + '_jet.png'
    output_fn = image.parent.joinpath(output_basename)
    # plt.imshow(im, vmin=0, vmax=1, cmap='jet')
    # plt.colorbar()
    print(output_fn)
    plt.imsave(output_fn, im, vmin=0, vmax=1, cmap='jet')
