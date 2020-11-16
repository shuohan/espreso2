"""Functions and classes to train the algorithm.

"""
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Iterable
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum

from pytorch_trainer.observer import SubjectObserver
from pytorch_trainer.train import Trainer
from pytorch_trainer.utils import NamedData
from pytorch_trainer.save import ThreadedSaver, ImageThread, SavePlot

from .config import Config
from .losses import GANLoss, SmoothnessLoss, CenterLoss, BoundaryLoss
from .utils import calc_fwhm


class SaveKernel(SavePlot):
    """Saves the kernel to a .png and a .npy files.

    """
    def save(self, filename, kernel):
        kernel = kernel.squeeze().numpy()
        self._save_plot(filename, kernel)
        self._save_npy(filename, kernel)

    def _save_npy(self, filename, kernel):
        filename = str(filename) + '.npy'
        np.save(filename, kernel)

    def _save_plot(self, filename, kernel):
        filename = str(filename) + '.png'
        fwhm, left, right = calc_fwhm(kernel)
        max_val = np.max(kernel)
        plt.cla()
        plt.plot(kernel, '-o')
        plt.plot([left, left], [0, max_val], 'k--')
        plt.plot([right, right], [0, max_val], 'k--')
        plt.text((left + right) / 2, max_val / 2, fwhm, ha='center')
        plt.grid(True)
        plt.tight_layout()
        plt.gcf().savefig(filename)


class KernelSaver(ThreadedSaver):
    """Saves the kernel after each epoch.

    """
    def __init__(self, dirname, step=100, save_init=False):
        super().__init__(dirname, save_init=save_init)
        self.step = step
        Path(self.dirname).mkdir(parents=True, exist_ok=True)

    def _init_thread(self):
        save_kernel = SaveKernel()
        return ImageThread(save_kernel, self.queue)

    def _check_subject_type(self, subject):
        assert isinstance(subject, TrainerHRtoLR)

    def update_on_epoch_end(self):
        if self.subject.epoch_ind % self.step == 0:
            self._save()

    def _save(self):
        kernel = self.subject.kernel_net.kernel
        pattern = 'epoch-%%0%dd' % len(str(self.subject.num_epochs))
        pattern = str(Path(self.dirname, pattern))
        filename = pattern % self.subject.epoch_ind
        self.queue.put(NamedData(filename, kernel))

        avg_kernel = self.subject.kernel_net.avg_kernel.cpu()
        pattern = 'avg_epoch-%%0%dd' % len(str(self.subject.num_epochs))
        pattern = str(Path(self.dirname, pattern))
        filename = pattern % self.subject.epoch_ind
        self.queue.put(NamedData(filename, avg_kernel))


class KernelEvaluator(SubjectObserver):
    """Evaluates the difference between the esitmated and the true kernels.

    """
    def __init__(self, true_kernel, kernel_length):
        super().__init__()
        self.kernel_length = kernel_length
        true_kernel = true_kernel.squeeze()
        left_pad = (self.kernel_length - len(true_kernel)) // 2
        right_pad = self.kernel_length - len(true_kernel) - left_pad
        true_kernel = np.pad(true_kernel, (left_pad, right_pad))
        self.true_kernel = torch.tensor(true_kernel)
        self.mse = np.nan

    def cuda(self):
        self.true_kernel = self.true_kernel.cuda()
        return self

    def update_on_batch_end(self):
        if self.epoch_ind % Config().eval_step == 0:
            self._calc_mae()
            self.notify_observers_on_batch_end()

    def update_on_epoch_end(self):
        if self.epoch_ind % Config().eval_step == 0:
            self.notify_observers_on_epoch_end()

    def _calc_mae(self):
        est_kernel = self.subject.kernel_net.avg_kernel.squeeze()
        with torch.no_grad():
            self.mae = F.l1_loss(self.true_kernel, est_kernel)

    @property
    def batch_size(self):
        return self.subject.batch_size

    @property
    def num_batches(self):
        return self.subject.num_batches

    @property
    def num_epochs(self):
        return self.subject.num_epochs

    @property
    def epoch_ind(self):
        return self.subject.epoch_ind

    @property
    def batch_ind(self):
        return self.subject.batch_ind


class TrainerHRtoLR(Trainer):
    """Trains GAN from high-resolution to low-resolution.

    Attributes:
        kernel_net (spest.network.KernelNet1d): The kernel estimation network.
        lr_disc (spest.network.LowResDiscriminator1d): The low-resolution
            patches discriminator.
        kn_optim (torch.optim.Optimizer): The :attr:`kernel_net` optimizer.
        lrd_optim (torch.optim.Optimizer): The :attr:`lr_dics` optimizer.
        dataloader (torch.nn.data.DataLoader): Yields image patches.
        scale_factor (iterable[float]): The upsampling scaling factor.
            It should be greater than 1.
        kn_gan_loss (torch.Tensor): The GAN loss for :attr:`kernel_net`.
        kn_tot_loss (torch.Tensor): The total loss for :attr:`kernel_net`.
        smoothness_loss (torch.Tensor): The kernel smoothness regularization.
        center_loss (torch.Tensor): The kernel center regularization.
        boundary_loss (torch.Tensor): The kernel boundary regularization.
        lrd_tot_loss (torch.Tensor): The total loss for :attr:`lr_disc`.

    """
    def __init__(self, kernel_net, lr_disc, kn_optim, lrd_optim, dataloader):
        super().__init__(Config().num_epochs)
        self.kernel_net = kernel_net
        self.lr_disc = lr_disc
        self.kn_optim = kn_optim
        self.lrd_optim = lrd_optim
        self.dataloader = dataloader
        self.scale_factor = Config().scale_factor

        self._gan_loss_func = GANLoss().cuda()
        self._smoothness_loss_func = SmoothnessLoss().cuda()
        self._center_loss_func = CenterLoss(Config().kernel_length).cuda()
        self._boundary_loss_func = BoundaryLoss(Config().kernel_length).cuda()

        self.kn_gan_loss = np.nan
        self.kn_gan_perm_loss = np.nan
        self.smoothness_loss = np.nan
        self.center_loss = np.nan
        self.boundary_loss = np.nan
        self.kn_tot_loss = np.nan
        self.lrd_tot_loss = np.nan

        self._kn_patch_cuda = None
        self._kn_patch_names = None
        self._kn_blur_cuda = None
        self._kn_alias_cuda = None
        self._kn_perm_cuda = None

        self._lrd_patch_cuda = None
        self._lrd_patch_names = None
        self._lrd_blur_cuda = None
        self._lrd_alias_cuda = None
        self._lrd_perm_cuda = None

        self._batch_ind = -1

    def get_model_state_dict(self):
        return {'kernel_net': self.kernel_net.state_dict(),
                'lr_disc': self.lr_disc.state_dict()}

    def get_optim_state_dict(self):
        return {'kn_optim': self.kn_optim.state_dict(),
                'lrd_optim': self.lrd_optim.state_dict()}

    def train(self):
        """Trains the algorithm."""
        self.notify_observers_on_train_start()
        for self._epoch_ind in range(self.num_epochs):
            self.notify_observers_on_epoch_start()

            self.notify_observers_on_batch_start()
            for batch in self.dataloader: # Note: len(dataloader) == 1
                self._lrd_patch_names = batch.name
                self._lrd_patch = batch.data
                self._lrd_patch_cuda = self._lrd_patch.cuda()
                self._train_lr_disc()

            if self.epoch_ind % Config().kn_update_step == 0:
                for batch in self.dataloader:
                    self._kn_patch_names = batch.name
                    self._kn_patch = batch.data
                    self._kn_patch_cuda = self._kn_patch.cuda()
                    self._train_kernel_net()
            self.notify_observers_on_batch_end()

            self.notify_observers_on_epoch_end()
        self.notify_observers_on_train_end()

    def _train_kernel_net(self):
        """Trains the generator :attr:`kernel_net`.
        
        """
        self.kn_optim.zero_grad()

        self._kn_blur_cuda = self.kernel_net(self._kn_patch_cuda)
        self._kn_alias_cuda = self._create_aliasing(self._kn_blur_cuda)
        self._kn_perm_cuda = self._kn_alias_cuda.permute(0, 1, 3, 2)
        self._lrd_pred_kn = self.lr_disc(self._kn_alias_cuda)
        self._lrd_pred_kn_perm = self.lr_disc(self._kn_perm_cuda)

        self.kn_gan_loss = -self._gan_loss_func(self._lrd_pred_kn, False)
        self.kn_gan_perm_loss = -self._gan_loss_func(self._lrd_pred_kn_perm, True)
        self.kn_tot_loss = 0.5 * (self.kn_gan_loss + self.kn_gan_perm_loss) + self._calc_reg()
        self.kn_tot_loss.backward()

        self.kn_optim.step()
        self.kernel_net.update_kernel()

    def _create_aliasing(self, patches):
        """Creates aliasing on patches."""
        down = [1 / self.scale_factor, 1]
        results = F.interpolate(patches, scale_factor=down, mode='bicubic')
        return results

    def _calc_reg(self):
        """Calculates kernel regularization."""
        kernel = self.kernel_net.kernel_cuda
        self.smoothness_loss = self._smoothness_loss_func(kernel)
        self.center_loss = self._center_loss_func(kernel)
        self.boundary_loss = self._boundary_loss_func(kernel)
        loss = Config().smoothness_loss_weight * self.smoothness_loss \
             + Config().center_loss_weight * self.center_loss \
             + Config().boundary_loss_weight * self.boundary_loss
        return loss

    def _train_lr_disc(self):
        """Trains the low-resolution discriminator :attr:`lr_disc`."""
        self.lrd_optim.zero_grad()
        with torch.no_grad():
            self._lrd_blur_cuda = self.kernel_net(self._lrd_patch_cuda)
            self._lrd_alias_cuda = self._create_aliasing(self._lrd_blur_cuda)
            self._lrd_perm_cuda = self._lrd_alias_cuda.permute(0, 1, 3, 2)
        self._lrd_pred_real = self.lr_disc(self._lrd_perm_cuda.detach())
        self._lrd_pred_fake = self.lr_disc(self._lrd_alias_cuda.detach())
        self._lrd_real_loss = self._gan_loss_func(self._lrd_pred_real, True)
        self._lrd_fake_loss = self._gan_loss_func(self._lrd_pred_fake, False)
        self.lrd_tot_loss = self._lrd_real_loss + self._lrd_fake_loss
        self.lrd_tot_loss.backward()
        self.lrd_optim.step()

    @property
    def num_batches(self):
        return len(self.dataloader)

    @property
    def batch_size(self):
        return self.dataloader.batch_size

    @property
    def batch_ind(self):
        return self._batch_ind + 1

    @property
    def lrd_patch(self):
        """Returns the current named lrd patches on CPU."""
        return NamedData(name=self._lrd_patch_names, data=self._lrd_patch.cpu())

    @property
    def lrd_blur(self):
        """Returns the current estimated blurred patches on CPU."""
        return self._lrd_blur_cuda.detach().cpu()

    @property
    def lrd_alias(self):
        """Returns the current estimated aliased patches on CPU."""
        return self._lrd_alias_cuda.detach().cpu()

    @property
    def lrd_perm(self):
        """Returns the current named low-resolution patches on CPU."""
        return self._lrd_perm_cuda.detach().cpu()

    @property
    def kn_patch(self):
        """Returns the current named kn patches on CPU."""
        return NamedData(name=self._kn_patch_names, data=self._kn_patch.cpu())

    @property
    def kn_blur(self):
        """Returns the current estimated blurred patches on CPU."""
        return self._kn_blur_cuda.detach().cpu()

    @property
    def kn_alias(self):
        """Returns the current estimated aliased patches on CPU."""
        return self._kn_alias_cuda.detach().cpu()

    @property
    def kn_perm(self):
        """Returns the current named low-resolution patches on CPU."""
        return self._kn_perm_cuda.detach().cpu()

    @property
    def lrd_pred_kn(self):
        """Returns the :attr:`lr_disc` output to update :attr:`kernel_net."""
        return self._lrd_pred_kn.detach().cpu()

    @property
    def lrd_pred_kn_perm(self):
        """Returns the :attr:`lr_disc` output to update :attr:`kernel_net."""
        return self._lrd_pred_kn_perm.detach().cpu()

    @property
    def lrd_pred_real(self):
        """Returns the :attr:`lr_disc` output of a true LR patch."""
        return self._lrd_pred_real.detach().cpu()

    @property
    def lrd_pred_fake(self):
        """Returns the :attr:`lr_disc` output of a generated patch."""
        return self._lrd_pred_fake.detach().cpu()


class InitKernelType(str, Enum):
    """The type of kernel to initialize to.

    Attributes:
        IMPULSE (str): Initialize to an impulse function.
        GAUSSIAN (str): Initialize to a Gaussian function.
        RECT (str): Initialize to a rect function.
        NONE (str): Do not initialize the kernel, i.e. the kernel is random.

    """
    IMPULSE = 'impulse'
    GAUSSIAN = 'guassian'
    RECT = 'rect'
    NONE = 'none'


def create_init_kernel(init_kernel_type, scale_factor, shape):
    """Creates the kernel to initialize.

    Args:
        init_kernel_type (str or InitKernelType): The type of the
            initialization.
        scale_factor (float): The scale factor  (greater than 1).
        shape (iterable[int]): The shape of the kernel.

    """
    init_kernel_type = InitKernelType(init_kernel_type)
    if init_kernel_type is InitKernelType.IMPULSE:
        kernel = torch.zeros(*shape, dtype=torch.float32)
        kernel[:, :, shape[2]//2, ...] = 1
    else:
        raise NotImplementedError
    return kernel


class TrainerKernelInit(Trainer):
    """Initializes the kernel using simulated HR and LR pairs.

    """
    def __init__(self, kernel_net, optim, dataloader, init_type='impulse'):
        super().__init__(Config().num_init_epochs)
        self.kernel_net = kernel_net
        self.optim = optim
        self.dataloader = dataloader
        self.init_type = init_type

        raise NotImplementedError

    def _init_kernel(self):
        """Initializes the kernel."""
        ref_kernel = self._create_init_kernel()
        self.init_optim.zero_grad()
        self._blur_cuda = self.kernel_net(self._hr_cuda)
        self._ref_cuda = F.conv2d(self._hr_cuda, ref_kernel)
        self.init_loss = self._init_loss_func(self._blur_cuda, self._ref_cuda)
        self.init_loss.backward()
        self.init_optim.step()

    def _create_init_kernel(self):
        """Creates the kernel to initialize to."""
        shape = list(self.kernel_net.impulse.shape)
        shape[2] = self.kernel_net.kernel_size
        kernel_type = self.init_kernel_type
        kernel = create_init_kernel(kernel_type, self.scale_factor, shape)
        return kernel.cuda()

    @property
    def ref(self):
        """Returns the blurred patches with a reference kernel on CPU."""
        return self._ref_cuda.detach().cpu()
