#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input image.')
parser.add_argument('-o', '--output', help='Output directory.')
parser.add_argument('-bs', '--batch-size', default=64, type=int,
                    help='The number of samples per mini-batch.')
parser.add_argument('-s', '--scale-factor', default=None, type=float,
                    help='Super resolution scale factor.')
parser.add_argument('-e', '--num-epochs', default=15000, type=int,
                    help='The number of epochs (iterations).')
parser.add_argument('-iss', '--image-save-step', default=5000, type=int,
                    help='The image saving step.')
parser.add_argument('-k', '--true-kernel', default=None)
parser.add_argument('-kl', '--kernel-length', default=21, type=int)
parser.add_argument('-lr', '--learning-rate', default=2e-4, type=float)
parser.add_argument('-na', '--no-aug', action='store_true')
parser.add_argument('-sk', '--symm-kernel', action='store_true')
parser.add_argument('-w', '--num-workers', default=0, type=int)
parser.add_argument('-sw', '--smoothness-loss-weight', default=1.0, type=float)
parser.add_argument('-z', '--z-axis', default=2, type=int)
parser.add_argument('-isz', '--image-save-zoom', default=1, type=int)
parser.add_argument('-wd', '--weight-decay', default=5e-2, type=float)
parser.add_argument('-lrdk', '--lrd-kernels', nargs='+', type=str,
                    default=((3, 1), (3, 1), (3, 1), (1, 1), (1, 1)),
                    help='Comma separated: 3,1 3,1 1,1.')
parser.add_argument('-lrdc', '--lrd-num-channels', default=(64, 64, 64, 64),
                    nargs='+', type=int)
parser.add_argument('-knc', '--kn-num-convs', default=3, type=int)
parser.add_argument('-knh', '--kn-num-channels', default=256, type=int)
parser.add_argument('-knk', '--kn-kernel-size', default=3, type=int)
parser.add_argument('-ns', '--num-epochs-per-stage', default=1, type=int)
parser.add_argument('-ps', '--patch-size', default=16, type=int)
parser.add_argument('-ie', '--num-init-epochs', default=100, type=int,
                    help='The number of init epochs (iterations).')
parser.add_argument('-zp', '--zero-pad-kn', action='store_true')
parser.add_argument('-in', '--intensity', default=1000.0, type=float)
parser.add_argument('-css', '--checkpoint-save-step', default=15000, type=int)
parser.add_argument('-c', '--checkpoint', default=None)
parser.add_argument('-d', '--debug', action='store_true')
args = parser.parse_args()


import os
import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
import warnings
from pytorchviz import make_dot

from sssrlib.patches import Patches, PatchesOr
from sssrlib.transform import Identity, Flip
from espreso2.config import Config
from espreso2.train import TrainerHRtoLR, KernelSaver, KernelEvaluator
from espreso2.train import TrainerKernelInit
from espreso2.networks import KernelNet, LowResDiscriminator, KernelNetZP
from espreso2.utils import calc_patch_size

from ptxl.log import DataQueue, TqdmEpochPrinter, EpochLogger, EpochPrinter
from ptxl.save import ImageSaver, CheckpointSaver


warnings.filterwarnings('ignore')

args.output = Path(args.output)
args.output.mkdir(parents=True, exist_ok=True)
im_output = args.output.joinpath('patches')
kernel_output = args.output.joinpath('kernel')
log_output = args.output.joinpath('loss.csv')
eval_log_output = args.output.joinpath('eval.csv')
config_output = args.output.joinpath('config.json')
arch_output = args.output.joinpath('arch')
checkpoint_output = args.output.joinpath('checkpoint')

xy = [0, 1, 2]
xy.remove(args.z_axis)
obj = nib.load(args.input)
image = obj.get_fdata(dtype=np.float32)

if args.scale_factor is None:
    zooms = np.round(obj.header.get_zooms(), 4).tolist()
    args.scale_factor = float(zooms[args.z_axis] / zooms[xy[0]])
    if not np.isclose(zooms[xy[0]], zooms[xy[1]]):
        raise RuntimeError('The resolutions of x and y are different.')
if args.scale_factor < 1:
    raise RuntimeError('Scale factor should be greater or equal to 1.')

if type(args.lrd_kernels[0]) is str:
    args.lrd_kernels = tuple(tuple(int(n) for n in lk.split(','))
                             for lk in args.lrd_kernels)

config = Config()
for key, value in args.__dict__.items():
    if hasattr(config, key):
        setattr(config, key, value)
config.input_image = os.path.abspath(str(args.input))
config.output_dirname = os.path.abspath(str(args.output))

image = image / image.max() * config.intensity

if args.debug:
    print('Image intensity range [{}, {}]'.format(image.min(), image.max()))

kn = KernelNet().cuda() if not config.zero_pad_kn else KernelNetZP().cuda()
lrd = LowResDiscriminator().cuda()
kn_optim = Adam(kn.parameters(), lr=config.learning_rate, betas=(0.5, 0.999),
                weight_decay=config.weight_decay)
lrd_optim = Adam(lrd.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))

if args.debug:
    x = torch.rand((1, 1, 64, 64)).float().cuda()
    kn_dot = make_dot(x, kn)
    kn_dot.render(arch_output.joinpath('kn'))
    lrd_dot = make_dot(x, lrd)
    lrd_dot.render(arch_output.joinpath('lrd'))

    with open(arch_output.joinpath('lrd.txt'), 'w') as f:
        f.write(lrd.__str__())
    with open(arch_output.joinpath('kn.txt'), 'w') as f:
        f.write(kn.__str__())

nz = image.shape[args.z_axis]
config.patch_size = calc_patch_size(config.patch_size, config.scale_factor, nz,
                                    kn.input_size_reduced)
config.weight_stride = (2, 2, 1)
config.save_json(config_output)

if args.debug:
    print(config)
    print(kn)
    print(lrd)
    print(kn_optim)
    print(lrd_optim)

# transforms = [] if args.no_aug else create_rot_flip()
transforms = [Identity(), Flip((0, )), Flip((2, )), Flip((0, 2))]

if args.debug:
    sample_weight_xz_gx_output = args.output.joinpath('sample_weights_xz_gx')
    sample_weight_xz_gz_output = args.output.joinpath('sample_weights_xz_gz')
    sample_weight_yz_gy_output = args.output.joinpath('sample_weights_yz_gy')
    sample_weight_yz_gz_output = args.output.joinpath('sample_weights_yz_gz')
else:
    sample_weight_xz_gx_output = None
    sample_weight_xz_gz_output = None
    sample_weight_yz_gy_output = None
    sample_weight_yz_gz_output = None

voxel_size = [zooms[xy[0]], zooms[xy[1]], zooms[args.z_axis]]
patch_size_xz = config.patch_size
patch_size_yz = np.array(config.patch_size)[[1, 0, 2]].tolist()

patches_xz_gx = Patches(patch_size_xz, image=image, x=xy[0], y=xy[1],
                        z=args.z_axis, transforms=transforms, sigma=1,
                        voxel_size=voxel_size, use_grads=[True, False, False],
                        weight_stride=config.weight_stride, avg_grad=False,
                        weight_dir=sample_weight_xz_gx_output,
                        compress=True, named=True, verbose=False).cuda()
patches_xz_gz = Patches(patch_size_xz, patches=patches_xz_gx, 
                        transforms=transforms, sigma=1,
                        voxel_size=voxel_size, use_grads=[False, False, True],
                        weight_stride=config.weight_stride, avg_grad=False,
                        weight_dir=sample_weight_xz_gz_output,
                        compress=True, named=True, verbose=False).cuda()

patches_yz_gy = Patches(patch_size_yz, patches=patches_xz_gx, 
                        transforms=transforms, sigma=1,
                        voxel_size=voxel_size, use_grads=[False, True, False],
                        weight_stride=config.weight_stride, avg_grad=False,
                        weight_dir=sample_weight_yz_gy_output,
                        compress=True, named=True, verbose=False).cuda()
patches_yz_gz = Patches(patch_size_yz, patches=patches_xz_gx, 
                        transforms=transforms, sigma=1,
                        voxel_size=voxel_size, use_grads=[False, False, True],
                        weight_stride=config.weight_stride, avg_grad=False,
                        weight_dir=sample_weight_yz_gz_output,
                        compress=True, named=True, verbose=False).cuda()

patches_xy = PatchesOr(patches_xz_gx, patches_yz_gy)
patches_z = PatchesOr(patches_xz_gz, patches_yz_gz)
loader_xy = patches_xy.get_dataloader(config.batch_size, num_workers=args.num_workers)
loader_z = patches_z.get_dataloader(config.batch_size, num_workers=args.num_workers)

if args.debug:
    print('Patches XY')
    print('----------')
    print(patches_xy)

    print('Patches Z')
    print('----------')
    print(patches_z)

trainer = TrainerHRtoLR(kn, lrd, kn_optim, lrd_optim, loader_xy, loader_z)
queue = DataQueue(['kn_gan_loss', 'smoothness_loss', 'center_loss',
                   'boundary_loss', 'kn_tot_loss', 'lrd_gan_loss'],
                  ['g_gan', 'smooth', 'center', 'bound', 'g_tot', 'desc_gan'])
if args.debug:
    printer = EpochPrinter(decimals=2, print_sep=False)
else:
    printer = TqdmEpochPrinter(decimals=2)

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
    # eval_printer = EpochPrinter(print_sep=False)
    eval_logger = EpochLogger(eval_log_output)
    # eval_queue.register(eval_printer)
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

checkpoint_saver = CheckpointSaver(checkpoint_output,
                                   step=config.checkpoint_save_step)

queue.register(printer)
queue.register(logger)
trainer.register(queue)
trainer.register(im_saver)
trainer.register(pred_saver)
trainer.register(kernel_saver)
trainer.register(checkpoint_saver)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    kn.load_state_dict(checkpoint['model_state_dict']['kernel_net'])
    lrd.load_state_dict(checkpoint['model_state_dict']['lr_disc'])
    kn_optim.load_state_dict(checkpoint['optim_state_dict']['kn_optim'])
    lrd_optim.load_state_dict(checkpoint['optim_state_dict']['lrd_optim'])
    trainer.set_epoch_ind(checkpoint['epoch'])
    print('Load checkpoint.')

if config.num_init_epochs > 0:
    # init_optim = Adam(kn.parameters(), lr=config.learning_rate,
    #                   betas=(0.5, 0.999), weight_decay=config.weight_decay)
    # print(init_optim)
    init_trainer = TrainerKernelInit(kn, kn_optim, loader_xy, init_type='impulse')

    init_im_output = args.output.joinpath('init_patches')
    init_kernel_output = args.output.joinpath('init_kernel')
    init_log_output = args.output.joinpath('init_loss.csv')


    if args.debug:
        init_queue = DataQueue(['init_loss'])
        init_printer = EpochPrinter(decimals=2, print_sep=False)
        init_logger = EpochLogger(init_log_output)
        init_queue.register(init_printer)
        init_queue.register(init_logger)

        init_kernel_saver = KernelSaver(init_kernel_output,
                                        step=config.image_save_step,
                                        save_init=True, truth=true_kernel)
        init_im_saver = ImageSaver(init_im_output,
                                   attrs=['patch', 'blur', 'ref_blur'],
                                   step=config.image_save_step,
                                   file_struct='epoch/sample', save_type='png_norm',
                                   save_init=False, prefix='patch',
                                   zoom=config.image_save_zoom, ordered=True)

        init_trainer.register(init_queue)
        init_trainer.register(init_im_saver)
        init_trainer.register(init_kernel_saver)

    init_trainer.train()

trainer.train()

result_npy = Path('kernel', 'avg_epoch-%d.npy' % args.num_epochs)
result_png = Path('kernel', 'avg_epoch-%d.png' % args.num_epochs)
args.output.joinpath('result.npy').symlink_to(result_npy)
args.output.joinpath('result.png').symlink_to(result_png)
