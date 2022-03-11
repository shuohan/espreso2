import argparse
from pathlib import Path
import sys
import warnings

import numpy as np

from .utils import calc_fwhm
from .train import TrainerBuilder


def train(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image-filename', help='Input image filename.', required=True)
    parser.add_argument('-o', '--output-dirname', help='Output directory.', required=True)
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='The number of samples per mini-batch.')
    parser.add_argument('-I', '--num-iters', default=2000, type=int,
                        help='The number of iterations.')
    parser.add_argument('-s', '--image-save-step', default=2000, type=int,
                        help='The image saving step.')
    parser.add_argument('-p', '--true-slice-profile', default=None)
    parser.add_argument('-L', '--slice-profile-length', default=21, type=int)
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('-N', '--warmup-learning-rate', default=1e-4, type=float)
    parser.add_argument('-z', '--z-axis', default=None, type=int)
    parser.add_argument('-Z', '--image-save-zoom', default=1, type=int)
    parser.add_argument('-d', '--sp-weight-decay', default=5e-3, type=float)
    parser.add_argument('-k', '--disc-kernel-sizes', nargs='+', type=str,
                        default=('3,1', '3,1', '3,1', '3,1', '3,1'),
                        help='Comma separated, e.g., 3,1 3,1 1,1.')
    parser.add_argument('-w', '--disc-nums-channels', default=(64, 64, 64, 64),
                        nargs='+', type=int)
    parser.add_argument('-a', '--disc-lrelu-neg-slope', default=0.1, type=float)
    parser.add_argument('-D', '--sp-num-convs', default=2, type=int)
    parser.add_argument('-W', '--sp-num-channels', default=256, type=int)
    parser.add_argument('-K', '--sp-kernel-size', default=3, type=int)
    parser.add_argument('-P', '--z-patch-size', default=16, type=int)
    parser.add_argument('-u', '--num-warmup-iters', default=80, type=int,
                        help='The number of warm-up iterations.')
    parser.add_argument('-M', '--sampler-mode', default='simple_foreground',
                        choices={'uniform', 'gradient', 'foreground',
                                 'simple_foreground', 'agg_foreground'})
    parser.add_argument('-e', '--eval-step', default=float('inf'), type=int)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-B', '--sp-avg-beta', default=0.99, type=float)
    parser.add_argument('-c', '--checkpoint', default=None)
    parser.add_argument('-C', '--checkpoint-save-step', default=15000, type=int)
    parser.add_argument('-g', '--debug', action='store_true')
    parser.add_argument('-U', '--augmentation', action='store_true')
    parser.add_argument('-m', '--no-symm-slice-profile', action='store_true')
    parser.add_argument('-t', '--boundary-loss-weight', default=10, type=float)
    parser.add_argument('-T', '--center-loss-weight', default=0, type=float)
    parser.add_argument('-O', '--smooth-loss-weight', default=0, type=float)
    parser.add_argument('-q', '--peak-loss-weight', default=1, type=float)
    parser.add_argument('-r', '--printer-mode', default='tqdm',
                        choices={'text', 'tqdm'})
    parser.add_argument('-E', '--weight-kernel-size', default=(4, 4, 1),
                        type=int, nargs='+')
    parser.add_argument('-R', '--weight-stride', default=(2, 2, 1),
                        type=int, nargs='+')
    parser.add_argument('-A', '--adam-betas', default=(0.5, 0.999), type=float,
                        nargs='+')

    parsed_args = parser.parse_args(sys.argv[1:] if args is None else args)

    warnings.filterwarnings('ignore')
    builder = TrainerBuilder(parsed_args)
    builder.build()
    if parsed_args.num_warmup_iters > 0:
        builder.warmup.train()
    builder.trainer.train()

    filename = 'iter-%d_sample-1_1_slice-profile' % parsed_args.num_iters
    result_npy = Path('slice_profiles', filename + '.npy')
    result_png = Path('slice_profiles', filename + '.png')
    Path(parsed_args.output_dirname).joinpath('result.npy').symlink_to(result_npy)
    Path(parsed_args.output_dirname).joinpath('result.png').symlink_to(result_png)


def fwhm(args=None):
    description = 'Calculate and print FWHM of a slice selection profile.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('input', help='The .npy file of the slice profile.')
    parsed_args = parser.parse_args(sys.argv[1:] if args is None else args)

    ssp = np.load(parsed_args.input)
    print(calc_fwhm(ssp)[0])
