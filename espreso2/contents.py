"""Values of tensors to record during training.

"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR

from ptxl.abstract import Contents
from ptxl.abstract import Observer
from ptxl.utils import Counter, Counters
from ptxl.log import Logger, Printer
from ptxl.log import TqdmPrinter as TqdmPrinter_
from ptxl.save import ImageSaver as ImageSaver_
from ptxl.save import SavePlot, create_save_image
from ptxl.save import CheckpointSaver as CheckpointSaver_

from .utils import calc_fwhm


class ContentsBuilder:
    """Builds a :class:`Contents` instance.

    Example:
        >>> builder = ContentsBuilder().build()
        >>> contents = builder.contents()

    Args:
        args (argparse.Namespace): The algorithm arguments.

    """
    def __init__(self, args):
        self.args = args
        self._contents = None

    @property
    def contents(self):
        return self._contents

    def build(self):
        """Builds the contents.

        Return:
            self.

        """
        self._counter = Counter('iter', self._get_num_iters())
        self._create_contents()
        self._set_observers()
        return self

    def _get_num_iters(self):
        raise NotImplementedError

    def _create_contents(self):
        raise NotImplementedError

    def _set_observers(self):
        self._create_printer()
        self._create_logger()
        self._create_checkpoint_saver()
        self._create_sp_saver()

    def _create_printer(self):
        attrs = self.contents.value_attrs
        if self.args.printer_mode == 'tqdm':
            printer = TqdmPrinter(decimals=2, attrs=attrs, name=self._get_name())
        elif self.args.printer_mode == 'text':
            printer = Printer(attrs=attrs)
        self.contents.register(printer)

    def _get_name(self):
        raise NotImplementedError

    def _create_logger(self):
        attrs = self.contents.value_attrs
        logger = Logger(self._get_log_filename(), attrs=attrs)
        self.contents.register(logger)

    def _get_log_filename(self):
        raise NotImplementedError

    def _create_sp_saver(self):
        attrs = self.contents.sp_attrs
        zoom = self.args.image_save_zoom
        save = SaveSliceProfile(self.args.true_slice_profile_values)
        saver = ImageSaver(self._get_slice_profile_dirname(), save,
                           attrs=attrs, step=self.args.image_save_step,
                           use_new_folder=False)
        self.contents.register(saver)

    def _get_slice_profile_dirname(self):
        raise NotImplementedError

    def _create_checkpoint_saver(self):
        saver = CheckpointSaver(self._get_checkpoint_dirname(),
                                step=self.args.checkpoint_save_step)
        self.contents.register(saver)

    def _get_checkpoint_dirname(self):
        raise NotImplementedError


class TrainContentsBuilder(ContentsBuilder):
    """Builds a :class:`TrainContents` instance.

    Args:
        sp_net (torch.nn.Module): The slice profile network.
        disc (torch.nn.Module): The discriminator.
        sp_optim (torch.nn.optim.Optimizer): The slice profile optimizer.
        disc_optim (torch.nn.optim.Optimizer): The discriminator optimizer.
        sp_sch (torch.nn.optim.lr_scheduler._LRScheduler): The slice profile
            learning rate scheduler.
        disc_sch (torch.nn.optim.lr_scheduler._LRScheduler): The discriminator
            learning rate scheduler.
        args (argparse.Namespace): The algorithm arguments.

    """
    def __init__(self, sp_net, disc, sp_optim, disc_optim, args):
        super().__init__(args)
        self.sp_net = sp_net
        self.disc = disc
        self.sp_optim = sp_optim
        self.disc_optim = disc_optim
        self.sch_builder = SchedulerBuilder(self.sp_optim, self.disc_optim,
                                            self.args.learning_rate,
                                            self.args.num_iters)
        self.args = args

    def _get_name(self):
        return 'train'

    def _get_num_iters(self):
        return self.args.num_iters

    def _create_contents(self):
        self._contents = TrainContents(self.sp_net, self.disc, self.sp_optim,
                                       self.disc_optim, self.sch_builder,
                                       self._counter)

    def _get_log_filename(self):
        return self.args.log_filename

    def _get_checkpoint_dirname(self):
        return self.args.output_checkpoint_dirname

    def _get_slice_profile_dirname(self):
        return self.args.output_slice_profile_dirname


class TrainContentsBuilderDebug(TrainContentsBuilder):
    """Builds a :class:`TrainContentsDebug` instance.

    """
    def _create_contents(self):
         cnt = TrainContentsDebug(self.sp_net, self.disc, self.sp_optim,
                                  self.disc_optim, self.sch_builder,
                                  self._counter)
         self._contents = cnt

    def _set_observers(self):
        if self.args.true_slice_profile:
            self._create_sp_evaluator()
        self._create_image_saver()
        self._create_prob_saver()
        super()._set_observers()

    def _create_image_saver(self):
        attrs = self.contents.im_attrs
        zoom = self.args.image_save_zoom
        save = create_save_image('png_norm', 'image', {'zoom': zoom})
        saver = ImageSaver(self.args.output_image_dirname, save,
                           attrs=attrs, step=self.args.image_save_step)
        self.contents.register(saver)

    def _create_prob_saver(self):
        attrs = self.contents.prob_attrs
        ind_offset = len(self.contents.im_attrs)
        zoom = self.args.image_save_zoom
        save = create_save_image('png', 'sigmoid', {'zoom': zoom})
        saver = ImageSaver(self.args.output_image_dirname, save,
                           ind_offset=ind_offset, attrs=attrs,
                           step=self.args.image_save_step)
        self.contents.register(saver)

    def _create_sp_evaluator(self):
        true_sp = self.args.true_slice_profile_values
        sp_eval = SliceProfileEvaluator(true_sp, self.args.slice_profile_length,
                                        self.args.eval_step)
        self.contents.register(sp_eval)


class WarmupContentsBuilder(ContentsBuilder):
    """Builds a :class:`WarmupContents` instance.

    Args:
        model (torch.nn.Module): The slice profile network.
        optim (torch.nn.Module): The slice profile optimizer.
        args (argparse.Namespace): The algorithm arguments.

    """
    def __init__(self, model, optim, args):
        self.model = model
        self.optim = optim
        self.args = args

    def _get_name(self):
        return 'warm-up'

    def _get_num_iters(self):
        return self.args.num_warmup_iters

    def _create_contents(self):
        self._contents = WarmupContents(self.model, self.optim, self._counter)

    def _get_log_filename(self):
        return self.args.warmup_log_filename

    def _get_checkpoint_dirname(self):
        return self.args.output_warmup_checkpoint_dirname

    def _get_slice_profile_dirname(self):
        return self.args.output_warmup_slice_profile_dirname


class SchedulerBuilder:
    def __init__(self, sp_optim, disc_optim, max_lr, total_steps):
        self.sp_optim = sp_optim
        self.disc_optim = disc_optim
        self.max_lr = max_lr
        self.total_steps = total_steps
        self._sp_sch = None
        self._disc_sch = None

    @property
    def sp_sch(self):
        return self._sp_sch

    @property
    def disc_sch(self):
        return self._disc_sch

    def build(self):
        self._sp_sch = OneCycleLR(self.sp_optim, max_lr=self.max_lr,
                                  total_steps=self.total_steps)
        self._disc_sch = OneCycleLR(self.disc_optim, max_lr=self.max_lr,
                                    total_steps=self.total_steps)
        return self


class TrainContents(Contents):
    """Records the contents during training.

    Attributes:
        sp_net (torch.nn.Module): The slice profile network.
        disc (torch.nn.Module): The discriminator.
        sp_optim (torch.nn.optim.Optimizer): The slice profile optimizer.
        disc_optim (torch.nn.optim.Optimizer): The discriminator optimizer.
        sp_sch (torch.nn.optim.lr_scheduler._LRScheduler): The slice profile
            learning rate scheduler.
        disc_sch (torch.nn.optim.lr_scheduler._LRScheduler): The discriminator
            learning rate scheduler.
        counter (ptxl.abstract.Counter): The iteration counter.

    """
    sp_attrs = ['slice_profile', 'avg_slice_profile']
    """list: Names of the slice profies to save."""

    im_attrs = ['disc_real', 'disc_real_blur', 'disc_real_down',
                'disc_real_down_t', 'disc_fake', 'disc_fake_blur',
                'disc_fake_down', 'sp_in', 'sp_blur', 'sp_down', 'sp_t_in',
                'sp_t_blur', 'sp_t_down', 'sp_t_down_t']
    """list: Names of the images to save."""

    prob_attrs = ['disc_real_prob', 'disc_fake_prob', 'sp_prob', 'sp_t_prob']
    """list: Names of the probability maps to save."""

    value_attrs = ['sp_adv_loss', 'sp_center_loss', 'sp_boundary_loss',
                   'sp_peak', 'sp_smooth_loss', 'sp_total_loss', 'disc_adv_loss', 'lr']
    """list: Names of the values to save."""

    def __init__(self, sp_net, disc, sp_optim, disc_optim, sch_builder, counter):
        self.sp_net = sp_net
        self.disc = disc
        self.sp_optim = sp_optim
        self.disc_optim = disc_optim
        self.sch_builder = sch_builder
        self.counter = counter

        self._values = dict()
        self._tensors_cpu = dict()
        self._tensors_cuda = dict()
        self._observers = list()

        for attr in self.im_attrs + self.prob_attrs + self.sp_attrs:
            self.set_tensor_cuda(attr, None, name=None)
        for attr in self.value_attrs:
            self.set_value(attr, float('nan'))

    def build_schedulers(self):
        self.sch_builder.build()
        self.sp_sch = self.sch_builder.sp_sch
        self.disc_sch = self.sch_builder.disc_sch

    def get_model_state_dict(self):
        return {'sp_net': self.sp_net.state_dict(),
                'disc': self.disc.state_dict()}

    def get_optim_state_dict(self):
        return {'sp_optim': self.sp_optim.state_dict(),
                'disc_optim': self.disc_optim.state_dict(),
                'sp_sch': self.sp_sch.state_dict(),
                'disc_sch': self.disc_sch.state_dict()}

    def load_state_dict(self, checkpoint):
        self.sp_net.load_state_dict(checkpoint['model_state_dict']['sp_net'])
        self.sp_optim.load_state_dict(checkpoint['optim_state_dict']['sp_optim'])
        self.disc.load_state_dict(checkpoint['model_state_dict']['disc'])
        self.disc_optim.load_state_dict(checkpoint['optim_state_dict']['disc_optim'])
        self.sp_sch.load_state_dict(checkpoint['optim_state_dict']['sp_sch'])
        self.disc_sch.load_state_dict(checkpoint['optim_state_dict']['disc_sch'])


class TrainContentsDebug(TrainContents):
    """Records the contents to save in debug mode."""
    value_attrs = TrainContents.value_attrs + ['sp_mae']
    """list: Names of the values to save."""


class WarmupContents(Contents):
    """Records the contents during warm-up.

    Attributes:
        model (torch.nn.Module): The slice profile network.
        optim (torch.nn.optim.Optimizer): The slice profile optimizer.
        counter (ptxl.abstract.Counter): The iteration counter.

    """
    sp_attrs = ['slice_profile', 'avg_slice_profile']
    """list: Names of the slice profies to save."""

    im_attrs = ['patches', 'blur', 'ref_blur']
    """list: Names of the images to save."""

    value_attrs = ['loss']
    """list: Names of the values to save."""

    def __init__(self, model, optim, counter):
        super().__init__(model, optim, counter)
        for attr in self.sp_attrs + self.im_attrs:
            self.set_tensor_cuda(attr, None, name=None)
        for attr in self.value_attrs:
            self.set_value(attr, float('nan'))


class ImageSaver(ImageSaver_):
    """Saves the images during training.

    """
    def _needs_to_update(self):
        rule1 = self.contents.counter.index1 % self.step == 0
        rule2 = self.contents.counter.has_reached_end()
        return rule1 or rule2


class CheckpointSaver(CheckpointSaver_):
    """Saves the checkpoints during training.

    """
    def _needs_to_update(self):
        rule1 = self.contents.counter.index1 % self.step == 0
        rule2 = self.contents.counter.has_reached_end()
        return rule1 or rule2

    def _get_counter_named_index(self):
        return self.contents.counter.named_index1

    def _get_counter_name(self):
        return self.contents.counter.name

    def _get_counter_index(self):
        return self.contents.counter.index1


class SaveSliceProfile(SavePlot):
    """Saves the slice profile to a .png and a .npy files.

    """
    def __init__(self, truth=None):
        super().__init__()
        self.truth = truth

    def save(self, filename, sp):
        sp = sp.squeeze().numpy()
        self._save_plot(filename, sp)
        self._save_npy(filename, sp)

    def _save_npy(self, filename, sp):
        filename = str(filename) + '.npy'
        np.save(filename, sp)

    def _save_plot(self, filename, sp):
        filename = str(filename) + '.png'
        fwhm, left, right = calc_fwhm(sp)
        max_val = np.max(sp)
        plt.cla()

        if self.truth is not None:
            plt.plot(self.truth, '-', color='tab:green')

        plt.plot(sp, '-o')
        plt.plot([left, right], [max_val / 2] * 2, 'x--', color='tab:red')
        plt.text((left + right) / 2, max_val / 4, fwhm, ha='center')

        plt.grid(True)
        plt.tight_layout()
        plt.gcf().savefig(filename)


class SliceProfileEvaluator(Observer):
    """Evaluates the difference between the esitmated and true slice profiles.

    """
    def __init__(self, true_sp, sp_length, step):
        super().__init__()
        self.sp_length = sp_length
        self.step = step
        self.true_sp = torch.tensor(true_sp).cuda()

    def _update(self):
        with torch.no_grad():
            est_sp = self.contents.sp_net.avg_slice_profile.squeeze()
            mae = F.l1_loss(self.true_sp, est_sp).item()
        self.contents.set_value('sp_mae', mae)

    def _needs_to_update(self):
        return self.contents.counter.index1 % self.step == 0


class TqdmPrinter(TqdmPrinter_):
    """Prints the warm-up/training progress.

    """
    def __init__(self, decimals=4, attrs=[], name=''):
        super().__init__(decimals=decimals, attrs=attrs)
        self.name = name

    def start(self):
        super().start()
        self._pbar.desc = self.name
