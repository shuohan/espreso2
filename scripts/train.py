#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image-filename', help='Input image filename.')
parser.add_argument('-o', '--output-dirname', help='Output directory.')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='The number of samples per mini-batch.')
parser.add_argument('-I', '--num-iters', default=15000, type=int,
                    help='The number of iterations.')
parser.add_argument('-s', '--image-save-step', default=5000, type=int,
                    help='The image saving step.')
parser.add_argument('-p', '--true-slice-profile', default=None)
parser.add_argument('-L', '--slice-profile-length', default=21, type=int)
parser.add_argument('-l', '--learning-rate', default=2e-4, type=float)
parser.add_argument('-z', '--z-axis', default=2, type=int)
parser.add_argument('-Z', '--image-save-zoom', default=1, type=int)
parser.add_argument('-d', '--sp-weight-decay', default=5e-2, type=float)
parser.add_argument('-k', '--disc-kernel-sizes', nargs='+', type=str,
                    default=('3,1', '3,1', '3,1', '1,1', '1,1'),
                    help='Comma separated, e.g., 3,1 3,1 1,1.')
parser.add_argument('-w', '--disc-nums-channels', default=(64, 64, 64, 64),
                    nargs='+', type=int)
parser.add_argument('-a', '--disc-lrelu-neg-slope', default=0.1, type=float)
parser.add_argument('-D', '--sp-num-convs', default=3, type=int)
parser.add_argument('-W', '--sp-num-channels', default=256, type=int)
parser.add_argument('-K', '--sp-kernel-size', default=3, type=int)
parser.add_argument('-P', '--z-patch-size', default=16, type=int)
parser.add_argument('-u', '--num-warmup-iters', default=100, type=int,
                    help='The number of warm-up iterations.')
parser.add_argument('-M', '--sampler-mode', default='uniform')
parser.add_argument('-e', '--eval-step', default=float('inf'), type=int)
parser.add_argument('-n', '--normalize', action='store_true')
parser.add_argument('-B', '--sp-avg-beta', default=0.99, type=float)
parser.add_argument('-c', '--checkpoint', default=None)
parser.add_argument('-C', '--checkpoint-save-step', default=15000, type=int)
parser.add_argument('-g', '--debug', action='store_true')
parser.add_argument('-N', '--no-augmentation', action='store_true')
parser.add_argument('-m', '--symm-slice-profile', action='store_true')
parser.add_argument('-t', '--boundary-loss-weight', default=10, type=float)
parser.add_argument('-T', '--center-loss-weight', default=1, type=float)
# parser.add_argument('-sw', '--smoothness-loss-weight', default=1.0, type=float)
# parser.add_argument('-zp', '--zero-pad-kn', action='store_true')

args = parser.parse_args()


import warnings
from espreso2.train import TrainerBuilder


warnings.filterwarnings('ignore')
builder = TrainerBuilder(args)
builder.build()
builder.warmup.train()
builder.trainer.train()

# result_npy = Path('kernel', 'avg_epoch-%d.npy' % args.num_epochs)
# result_png = Path('kernel', 'avg_epoch-%d.png' % args.num_epochs)
# args.output.joinpath('result.npy').symlink_to(result_npy)
# args.output.joinpath('result.png').symlink_to(result_png)
