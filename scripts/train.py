#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input image.')
parser.add_argument('-o', '--output', help='Output directory.')
parser.add_argument('-bs', '--batch-size', default=32, type=int,
                    help='The number of samples per mini-batch.')
parser.add_argument('-s', '--scale-factor', default=None, type=float,
                    help='Super resolution scale factor.')
parser.add_argument('-e', '--num-epochs', default=10000, type=int,
                    help='The number of epochs (iterations).')
parser.add_argument('-iss', '--image-save-step', default=100, type=int,
                    help='The image saving step.')
parser.add_argument('-k', '--true-kernel', default=None)
parser.add_argument('-kl', '--kernel-length', default=21, type=int)
parser.add_argument('-lr', '--learning-rate', default=2e-4, type=float)
parser.add_argument('-na', '--no-aug', action='store_true')
parser.add_argument('-w', '--num-workers', default=0, type=int)
parser.add_argument('-sw', '--smoothness-loss-weight', default=1.0, type=float)
parser.add_argument('-z', '--z-axis', default=2, type=int)
parser.add_argument('-isz', '--image-save-zoom', default=1, type=int)
parser.add_argument('-wd', '--weight-decay', default=0, type=float)
parser.add_argument('-lrdk', '--lrd-kernels', nargs='+', type=str,
                    default=((3, 1), (3, 1), (3, 1), (3, 1), (3, 1)),
                    help='Comma separated: 3,1 3,1 1,1.')
parser.add_argument('-lrdc', '--lrd-num-channels', default=(64, 128, 256, 512),
                    nargs='+', type=int)
parser.add_argument('-knc', '--kn-num-convs', default=6, type=int)
parser.add_argument('-ns', '--num-epochs-per-stage', default=1000, type=int)
parser.add_argument('-ps', '--patch-size', default=7, type=int)
args = parser.parse_args()


import os
import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
import warnings

from sssrlib.patches import Patches, PatchesOr
from sssrlib.transform import create_rot_flip
from spest.config import Config
from spest.train import TrainerHRtoLR, KernelSaver, KernelEvaluator
from spest.networks import KernelNet, LowResDiscriminator
from spest.utils import calc_patch_size

from pytorch_trainer.log import DataQueue, EpochPrinter, EpochLogger
from pytorch_trainer.save import ImageSaver


warnings.filterwarnings("ignore")

args.output = Path(args.output)
args.output.mkdir(parents=True, exist_ok=True)
im_output = args.output.joinpath('patches')
kernel_output = args.output.joinpath('kernel')
log_output = args.output.joinpath('loss.csv')
eval_log_output = args.output.joinpath('evseg_segal_loss.csv')
config_output = args.output.joinpath('config.json')

xy = [0, 1, 2]
xy.remove(args.z_axis)
obj = nib.load(args.input)
image = obj.get_fdata(dtype=np.float32)
if args.scale_factor is None:
    zooms = obj.header.get_zooms()
    args.scale_factor = float(zooms[args.z_axis] / zooms[xy[0]])
    if zooms[xy[0]] != zooms[xy[1]] and not args.no_aug:
        raise RuntimeError('The resolutions of x and y are different.')
if args.scale_factor < 1:
    raise RuntimeError('Scale factor should be greater or equal to 1.')

args.lrd_kernels = tuple(tuple(int(n) for n in lk.split(','))
                         for lk in args.lrd_kernels)

config = Config()
for key, value in args.__dict__.items():
    if hasattr(config, key):
        setattr(config, key, value)
config.add_config('input_image', os.path.abspath(str(args.input)))
config.add_config('output_dirname', os.path.abspath(str(args.output)))

kn = KernelNet().cuda()
lrd = LowResDiscriminator().cuda()
kn_optim = Adam(kn.parameters(), lr=config.learning_rate, betas=(0.5, 0.999),
                weight_decay=config.weight_decay)
lrd_optim = Adam(lrd.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))

nz = image.shape[args.z_axis]
config.patch_size = calc_patch_size(config.patch_size, config.scale_factor, nz,
                                    kn.input_size_reduced)
weight_stride = [8, 8, int(max(config.scale_factor // 8, 1))]
config.add_config('weight_stride', weight_stride)
# config.add_config('weight_stride', (1, 1, 1))
print(config)
config.save_json(config_output)

print(kn)
print(lrd)
print(kn_optim)
print(lrd_optim)

# transforms = [] if args.no_aug else create_rot_flip()
patches = Patches(image, config.patch_size, x=xy[0], y=xy[1], z=args.z_axis,
                  named=True, weight_stride=config.weight_stride,
                  avg_grad=False).cuda()
dataloader = patches.get_dataloader(config.batch_size, args.num_workers)
print('Patches')
print('----------')
print(patches)

trainer = TrainerHRtoLR(kn, lrd, kn_optim, lrd_optim, dataloader)
queue = DataQueue(['kn_gan_loss', 'smoothness_loss', 'center_loss',
                   'boundary_loss', 'kn_tot_loss', 'lrd_gan_loss'])
printer = EpochPrinter(print_sep=False, decimals=2)
logger = EpochLogger(log_output)

true_kernel = None
if args.true_kernel is not None:
    true_kernel = np.load(args.true_kernel)
    true_kernel = true_kernel.squeeze()
    left_pad = (config.kernel_length - len(true_kernel)) // 2
    right_pad = config.kernel_length - len(true_kernel) - left_pad
    true_kernel = np.pad(true_kernel, (left_pad, right_pad))
    evaluator = KernelEvaluator(true_kernel, config.kernel_length).cuda()
    eval_queue = DataQueue(['mae'])
    eval_printer = EpochPrinter(print_sep=False)
    eval_logger = EpochLogger(eval_log_output)
    eval_queue.register(eval_printer)
    eval_queue.register(eval_logger)
    evaluator.register(eval_queue)
    trainer.register(evaluator)

attrs = ['lrd_real', 'lrd_real_blur', 'lrd_real_alias', 'lrd_real_alias_t',
         'lrd_fake', 'lrd_fake_blur', 'lrd_fake_alias',
         'kn_in', 'kn_blur', 'kn_alias',
         'kn_t_in', 'kn_t_blur', 'kn_t_alias', 'kn_t_alias_t']
im_saver = ImageSaver(im_output, attrs=attrs, step=config.image_save_step,
                      file_struct='epoch/sample', save_type='png_norm',
                      save_init=False, prefix='patch',
                      zoom=config.image_save_zoom, ordered=True)

attrs = ['lrd_real_prob', 'lrd_fake_prob', 'kn_prob', 'kn_t_prob']
pred_saver = ImageSaver(im_output, attrs=attrs, step=config.image_save_step,
                        file_struct='epoch/sample', save_type='png',
                        image_type='sigmoid', save_init=False, prefix='lrd',
                        zoom=config.image_save_zoom, ordered=True)

kernel_saver = KernelSaver(kernel_output, step=config.image_save_step,
                           save_init=True, truth=true_kernel)

queue.register(printer)
queue.register(logger)
trainer.register(queue)
trainer.register(im_saver)
trainer.register(pred_saver)
trainer.register(kernel_saver)

trainer.train()
