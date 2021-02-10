#!/usr/bin/env python

import argparse

description = 'Calculate and print FWHM of a slice selection profile.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('input', help='The .npy file of the slice profile.')
args = parser.parse_args()


import numpy as np
from espreso2.utils import calc_fwhm


ssp = np.load(args.input)
fwhm = calc_fwhm(ssp)[0]
print(fwhm)
