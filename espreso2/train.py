"""Functions and classes to train the algorithm.

"""
import json
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.optim import Adam
from pathlib import Path
from enum import Enum
from scipy.signal import gaussian

from .losses import GANLoss, SmoothnessLoss, CenterLoss, BoundaryLoss
from .contents import TrainContentsBuilder, TrainContentsBuilderDebug
from .contents import WarmupContentsBuilder
from .sample import SamplerBuilderUniform, SamplerType
from .sample import SamplerBuilderGrad, SamplerBuilderGrad, SamplerBuilderFG
from .networks import SliceProfileNet, Discriminator


class TrainerBuilder:
    def __init__(self, args):
        self.args = args
        self._trainer = None
        self._warmup = None

    @property
    def trainer(self):
        return self._trainer

    @property
    def warmup(self):
        return self._warmup

    def build(self):
        self._specify_outputs()
        self._parse_image()
        self._create_sp_net()
        self._create_disc_net()
        self._create_sp_optim()
        self._create_disc_optim()
        self._load_true_slice_profile()
        self._create_warmup_contents()
        self._create_train_contents()
        self._calc_patch_size()
        self._create_samplers()
        self._create_warmup()
        self._create_trainer()
        self._save_args()

    def _specify_outputs(self):
        output_dirname = Path(self.args.output_dirname)
        output_dirname.mkdir(parents=True)
        self.args.output_image_dirname \
            = str(output_dirname.joinpath('patches'))
        self.args.output_warmup_slice_profile_dirname \
            = str(output_dirname.joinpath('patches_warmup'))
        self.args.output_slice_profile_dirname \
            = str(output_dirname.joinpath('slice_profiles'))
        self.args.output_warmup_slice_profile_dirname \
            = str(output_dirname.joinpath('slice_profiles_warmup'))
        self.args.log_filename \
            = str(output_dirname.joinpath('loss.csv'))
        self.args.warmup_log_filename \
            = str(output_dirname.joinpath('loss_warmup.csv'))
        self.args.config_filename \
            = str(output_dirname.joinpath('config.json'))
        self.args.output_checkpoint_dirname \
            = str(output_dirname.joinpath('checkpoints'))
        self.args.output_warmup_checkpoint_dirname \
            = str(output_dirname.joinpath('checkpoints_warmup'))

    def _create_sp_net(self):
        self._sp_net = SliceProfileNet(num_channels=self.args.sp_num_channels,
                                       kernel_size=self.args.sp_kernel_size,
                                       num_convs=self.args.sp_num_convs,
                                       sp_length=self.args.slice_profile_length,
                                       sp_avg_beta=self.args.sp_avg_beta).cuda()

    def _create_disc_net(self):
        ks = tuple(tuple(int(k) for k in dk.split(',')) for
                   dk in self.args.disc_kernel_sizes)
        self.args.disc_kernel_sizes = ks
        disc = Discriminator(nums_channels=self.args.disc_nums_channels,
                             kernel_sizes=self.args.disc_kernel_sizes,
                             lrelu_neg_slope=self.args.disc_lrelu_neg_slope)
        self._disc = disc.cuda()

    def _create_sp_optim(self):
        self._sp_optim = Adam(self._sp_net.parameters(),
                              lr=self.args.learning_rate,
                              betas=(0.5, 0.999),
                              weight_decay=self.args.sp_weight_decay)

    def _create_disc_optim(self):
        self._disc_optim = Adam(self._disc.parameters(),
                                lr=self.args.learning_rate,
                                betas=(0.5, 0.999))

    def _parse_image(self):
        self._nifti = nib.load(self.args.image_filename)
        self._image = self._nifti.get_fdata(dtype=np.float32)
        if self.args.normalize:
            self._image = self._image / self._image.max() * 1000
        voxel_size = self._nifti.header.get_zooms()
        self.args.voxel_size = np.round(voxel_size, 4).astype(float).tolist()
        self._get_axis_order()
        x_vs = self.args.voxel_size[self.args.x_axis]
        y_vs = self.args.voxel_size[self.args.y_axis]
        z_vs = self.args.voxel_size[self.args.z_axis]
        self.args.scale_factor = float(x_vs) / float(z_vs)
        if not np.isclose(x_vs, y_vs):
            raise RuntimeError('The resolutions of x and y are different.')
        if self.args.scale_factor > 1:
            raise RuntimeError('Scale factor should be less or equal to 1.')

    def _get_axis_order(self):
        z = np.argmax(self.args.voxel_size)
        xy = list(range(len(self.args.voxel_size)))
        xy.remove(z)
        self.args.x_axis = int(xy[0])
        self.args.y_axis = int(xy[1])
        self.args.z_axis = int(z)

    def _calc_patch_size(self):
        nz = self._image.shape[self.args.z_axis]
        scale_factor = 1 / self.args.scale_factor
        reduced = self._sp_net.input_size_reduced
        lr_patch_size = np.minimum(self.args.z_patch_size, nz)
        image = torch.rand(1, 1, lr_patch_size, lr_patch_size).float()
        up = F.interpolate(image, scale_factor=scale_factor, mode='bicubic')
        hr_patch_size = int(up.shape[2])
        down = F.interpolate(up, scale_factor=1/scale_factor, mode='bicubic')
        lr_patch_size = int(down.shape[2])
        self.args.patch_size = (hr_patch_size + reduced, 1, lr_patch_size)

    def _load_true_slice_profile(self):
        if self.args.debug and self.args.true_slice_profile:
            sp = np.load(self.args.true_slice_profile)
            assert self.args.slice_profile_length >= len(sp)
            left_pad = (self.args.slice_profile_length - len(sp)) // 2
            right_pad = self.args.slice_profile_length - len(sp) - left_pad
            sp = np.pad(sp, (left_pad, right_pad))
            self.args.true_slice_profile_values = sp.tolist()

    def _create_warmup_contents(self):
        builder = WarmupContentsBuilder(self._sp_net, self._sp_optim, self.args)
        self._warmup_contents = builder.build().contents

    def _create_train_contents(self):
        if self.args.debug:
            Builder = TrainContentsBuilderDebug
        else:
            Builder = TrainContentsBuilder
        builder = Builder(self._sp_net, self._disc, self._sp_optim,
                          self._disc_optim, self.args)
        self._train_contents = builder.build().contents

    def _create_samplers(self):
        stype = SamplerType(self.args.sampler_mode)
        if stype is SamplerType.UNIFORM:
            builder = SamplerBuilderUniform(self.args.patch_size, self._image,
                                            self.args.x_axis, self.args.y_axis,
                                            self.args.z_axis,
                                            self.args.voxel_size)
            builder.build()
            self._xy_sampler = builder.sampler
            self._z_sampler = builder.sampler
        elif stype is SamplerType.GRADIENT:
            builder = SamplerBuilderGrad(self.args.patch_size, self._image,
                                         self.args.x_axis, self.args.y_axis,
                                         self.args.z_axis, self.args.voxel_size,
                                         (4, 4, 1), (2, 2, 1))
            builder.build()
            self._xy_sampler = builder.sampler_xy
            self._z_sampler = builder.sampler_z

        elif stype is SamplerType.FOREGROUND:
            builder = SamplerBuilderFG(self.args.patch_size, self._image,
                                       self.args.x_axis, self.args.y_axis,
                                       self.args.z_axis, self.args.voxel_size)
            builder.build()
            self._xy_sampler = builder.sampler
            self._z_sampler = builder.sampler

    def _create_trainer(self):
        self._trainer = Trainer(self._train_contents, self._xy_sampler,
                                self._z_sampler, self.args.scale_factor,
                                self.args.batch_size,
                                self.args.boundary_loss_weight,
                                self.args.center_loss_weight)

    def _create_warmup(self):
        ref_sp = create_warmup_sp('impulse', self.args.slice_profile_length)
        self._warmup = Warmup(self._warmup_contents, self._xy_sampler,
                              ref_sp.cuda(), self.args.batch_size)

    def _save_args(self):
        result = dict()
        for arg in vars(self.args):
            result[arg] = getattr(self.args, arg)
        with open(self.args.config_filename, 'w') as jfile:
            json.dump(result, jfile, indent=4)


class _Trainer:
    def __init__(self, contents, batch_size):
        self.contents = contents
        self.batch_size = batch_size

    def train(self):
        self._sample_patch_indices()
        self.contents.start_observers()
        for i in self.contents.counter:
            self._train()
            self.contents.notify_observers()
        self.contents.close_observers()

    def _train(self):
        raise NotImplementedError

    def _sample_patch_indices(self):
        raise NotImplementedError

    def _sample_patch_indices_from(self, sampler):
        num_iters = self.contents.counter.num
        num_indices = self.batch_size * num_iters
        return sampler.sample_indices(num_indices)

    def _sample_patches(self, indices, sampler):
        iter = self.contents.counter.index0
        start_ind = iter * self.batch_size
        stop_ind = start_ind + self.batch_size
        sub_indices = indices[start_ind : stop_ind]
        patches = sampler.get_patches(sub_indices)
        return patches


class Trainer(_Trainer):
    def __init__(self, contents, xy_sampler, z_sampler, scale_factor,
                 batch_size, boundary_loss_weight, center_loss_weight):
        self.contents = contents
        self.xy_sampler = xy_sampler
        self.z_sampler = z_sampler
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.boundary_loss_weight = boundary_loss_weight
        self.center_loss_weight = center_loss_weight
        self._init_loss_funcs()

    def _init_loss_funcs(self):
        self._gan_loss_func = GANLoss().cuda()
        sp_length = self.contents.sp_net.sp_length
        self._center_loss_func = CenterLoss(sp_length).cuda()
        self._boundary_loss_func = BoundaryLoss(sp_length).cuda()

    def _train(self):
        self._train_disc()
        self._train_sp_net()

    def _sample_patch_indices(self):
        self._sp_xy_indices = self._sample_patch_indices_from(self.xy_sampler)
        self._sp_z_indices = self._sample_patch_indices_from(self.z_sampler)
        self._disc_xy_indices = self._sample_patch_indices_from(self.xy_sampler)
        self._disc_z_indices = self._sample_patch_indices_from(self.z_sampler)

    def _train_disc(self):
        self.contents.disc_optim.zero_grad()
        disc_fake = self._sample_patches(self._disc_xy_indices, self.xy_sampler)
        disc_real = self._sample_patches(self._disc_z_indices, self.z_sampler)

        with torch.no_grad():
            disc_fake_blur = self.contents.sp_net(disc_fake.data)
            disc_fake_down = self._downsample(disc_fake_blur)
            disc_real_blur = self.contents.sp_net(disc_real.data)
            disc_real_down = self._downsample(disc_real_blur)
            disc_real_down_t = disc_real_down.permute(0, 1, 3, 2)
        disc_fake_prob = self.contents.disc(disc_fake_down.detach())
        disc_real_prob = self.contents.disc(disc_real_down_t.detach())

        disc_adv_loss = self._calc_adv_loss(disc_fake_prob, disc_real_prob)
        disc_adv_loss.backward()
        self.contents.disc_optim.step()

        self.contents.set_tensor_cuda('disc_fake', disc_fake.data, disc_fake.name)
        self.contents.set_tensor_cuda('disc_fake_blur', disc_fake_blur, disc_fake.name)
        self.contents.set_tensor_cuda('disc_fake_down', disc_fake_down, disc_fake.name)
        self.contents.set_tensor_cuda('disc_fake_prob', disc_fake_prob, disc_fake.name)

        self.contents.set_tensor_cuda('disc_real', disc_real.data, disc_real.name)
        self.contents.set_tensor_cuda('disc_real_blur', disc_real_blur, disc_real.name)
        self.contents.set_tensor_cuda('disc_real_down', disc_real_down, disc_real.name)
        self.contents.set_tensor_cuda('disc_real_down_t', disc_real_down_t, disc_real.name)
        self.contents.set_tensor_cuda('disc_real_prob', disc_real_prob, disc_real.name)

        self.contents.set_value('disc_adv_loss', disc_adv_loss.item())

    def _train_sp_net(self):
        self.contents.sp_optim.zero_grad()
        sp_in = self._sample_patches(self._sp_xy_indices, self.xy_sampler)
        sp_t_in = self._sample_patches(self._sp_z_indices, self.z_sampler)

        sp_blur = self.contents.sp_net(sp_in.data)
        sp_down = self._downsample(sp_blur)
        sp_prob = self.contents.disc(sp_down)

        sp_t_blur = self.contents.sp_net(sp_t_in.data)
        sp_t_down = self._downsample(sp_t_blur)
        sp_t_down_t = sp_t_down.permute(0, 1, 3, 2)
        sp_t_prob = self.contents.disc(sp_t_down_t)

        loss = self._calc_sp_loss(sp_prob, sp_t_prob)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.contents.sp_net.parameters(), 1)
        self.contents.sp_optim.step()
        self.contents.sp_net.update_slice_profile()

        self.contents.set_tensor_cuda('sp_in', sp_in.data, sp_in.name)
        self.contents.set_tensor_cuda('sp_blur', sp_blur, sp_in.name)
        self.contents.set_tensor_cuda('sp_down', sp_down, sp_in.name)
        self.contents.set_tensor_cuda('sp_prob', sp_prob, sp_in.name)

        self.contents.set_tensor_cuda('sp_t_in', sp_t_in.data, sp_t_in.name)
        self.contents.set_tensor_cuda('sp_t_blur', sp_t_blur, sp_t_in.name)
        self.contents.set_tensor_cuda('sp_t_down', sp_t_down, sp_t_in.name)
        self.contents.set_tensor_cuda('sp_t_down_t', sp_t_down_t, sp_t_in.name)
        self.contents.set_tensor_cuda('sp_t_prob', sp_t_prob, sp_t_in.name)

        sp = self.contents.sp_net.slice_profile
        avg_sp = self.contents.sp_net.avg_slice_profile
        self.contents.set_tensor_cuda('slice_profile', sp.detach())
        self.contents.set_tensor_cuda('avg_slice_profile', avg_sp)

    def _downsample(self, patches):
        sf = [self.scale_factor, 1]
        results = F.interpolate(patches, scale_factor=sf, mode='bicubic')
        return results

    def _calc_adv_loss(self, fake_prob, real_prob):
        fake_loss = self._gan_loss_func(fake_prob, False)
        real_loss = self._gan_loss_func(real_prob, True)
        return (fake_loss + real_loss) / 2

    def _calc_sp_loss(self, sp_prob, sp_t_prob):
        sp_adv_loss = -self._calc_adv_loss(sp_prob, sp_t_prob)

        sp = self.contents.sp_net.slice_profile
        sp_center_loss = self._center_loss_func(sp)
        sp_boundary_loss = self._boundary_loss_func(sp)

        sp_total_loss = sp_adv_loss \
            + self.center_loss_weight * sp_center_loss \
            + self.boundary_loss_weight * sp_boundary_loss

        self.contents.set_value('sp_adv_loss', sp_adv_loss.item())
        self.contents.set_value('sp_center_loss', sp_center_loss.item())
        self.contents.set_value('sp_boundary_loss', sp_boundary_loss.item())
        self.contents.set_value('sp_total_loss', sp_total_loss.item())

        # if self.epoch_ind <= Config().smoothness_loss_epochs:
        #     self.smoothness_loss = self._smoothness_loss_func(kernel)
        #     loss += Config().smoothness_loss_weight * self.smoothness_loss

        return sp_total_loss


class WarmupSPType(str, Enum):
    """The type of kernel to initialize to.

    Attributes:
        IMPULSE (str): Initialize to an impulse function.
        GAUSSIAN (str): Initialize to a Gaussian function.
        RECT (str): Initialize to a rect function.
        NONE (str): Do not initialize the kernel, i.e. the kernel is random.

    """
    IMPULSE = 'impulse'
    GAUSSIAN = 'gaussian'
    RECT = 'rect'
    NONE = 'none'


def create_warmup_sp(warmup_sp_type, sp_length, fwhm=None):
    """Creates the slice profile for warm-up.

    Args:
        warmup_up_type (str or WarmupSPType): The type of the warmup profile.
        sp_length (int): The length of the slice profile.
        fwhm (float): The full width at half maximum for Gaussian.

    """
    warmup_sp_type = WarmupSPType(warmup_sp_type)
    if warmup_sp_type is WarmupSPType.IMPULSE:
        sp = torch.zeros([1, 1, sp_length, 1], dtype=torch.float32)
        sp[:, :, sp_length//2, ...] = 1
    elif warmup_sp_type is WarmupSPType.GAUSSIAN:
        sp = gaussian(sp_length, fwhm / 2.355, sym=True)
        sp = torch.tensor(sp).float()[None, None, :, None]
        sp = sp / torch.sum(sp)
    return sp


class Warmup(_Trainer):
    """Learning warm-up.

    The main effect is to accumulate optimizer momentum towards a "bell-shaped"
    slice profile estimation.

    """
    def __init__(self, contents, sampler, ref_sp, batch_size):
        self.contents = contents
        self.sampler = sampler
        self.ref_sp = ref_sp
        self.batch_size = batch_size
        self._loss_func = torch.nn.MSELoss()

    def _sample_patch_indices(self):
        self._indices = self._sample_patch_indices_from(self.sampler)

    def _train(self):
        self.contents.optim.zero_grad()

        patches = self._sample_patches(self._indices, self.sampler)
        blur = self.contents.model(patches.data)
        ref_blur = F.conv2d(patches.data, self.ref_sp)

        loss = self._loss_func(blur, ref_blur)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.contents.sp_net.parameters(), 1)
        self.contents.optim.step()
        self.contents.model.update_slice_profile()

        self.contents.set_tensor_cuda('patches', patches.data, patches.name)
        self.contents.set_tensor_cuda('blur', blur, patches.name)
        self.contents.set_tensor_cuda('ref_blur', ref_blur, patches.name)

        sp = self.contents.model.slice_profile
        avg_sp = self.contents.model.avg_slice_profile
        self.contents.set_tensor_cuda('slice_profile', sp.detach())
        self.contents.set_tensor_cuda('avg_slice_profile', avg_sp)

        self.contents.set_value('loss', loss.item())
