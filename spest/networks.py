import torch
import math
from torch import nn
import torch.nn.functional as F

from .config import Config


class _KernelNet(nn.Sequential):
    """Abstract class for kernel net.

    """
    def __init__(self):
        super().__init__()
        config = Config()

        ks = 3
        num_ch = config.kn_num_channels
        num_convs = config.kn_num_convs

        size = self._calc_input_weight_size(ks)
        self.input_weight = nn.Parameter(torch.zeros(1, num_ch, size, 1))
        self.reset_parameters()

        for i in range(num_convs - 1):
            self.add_module('conv%d' % i, self._create_conv(num_ch, ks))
            self.add_module('relu%d' % i, nn.ReLU())
        self.add_module('conv%d' % (num_convs - 1), self._create_conv(1, ks))

        # Calling self._calc_kernel() at init cannot put tensors into cuda
        self._kernel_cuda = None
        self.register_buffer('avg_kernel', self._calc_kernel().detach())

    def _calc_input_weight_size(self, ks):
        raise NotImplementedError

    def _create_conv(self, out_channels, kernel_size):
        raise NotImplementedError

    def _calc_kernel(self):
        """Calculates the current kernel with shape ``[1, 1, length]``."""
        kernel = self.input_weight
        for module in self:
            kernel = module(kernel)
        if Config().symm_kernel:
            kernel = self._make_symmetric_kernel(kernel)
        kernel = F.softmax(kernel, dim=2)
        return kernel

    def _make_symmetric_kernel(self, kernel):
        return 0.5 * (kernel + torch.flip(kernel, (2, )))

    def update_kernel(self):
        r"""Updates the current kernel and calculates the moving average.

        This method updates the moving average :math:`k_t` as:

        .. math::

            \bar{k}_t = (1 - \beta) k_t + \beta \bar{k}_{t-1}

        Note:
            This function should be called at each iteration after
            backpropagation.

        """
        self._kernel_cuda = self._calc_kernel()
        beta = Config().kernel_avg_beta
        self.avg_kernel = (1 - beta) * self.kernel_cuda.detach() \
            + beta * self.avg_kernel

    @property
    def kernel_cuda(self):
        """Returns the current kernel on CUDA with shape ``[1, kernel_len]``."""
        if self._kernel_cuda is None:
            self._kernel_cuda = self._calc_kernel()
        return self._kernel_cuda

    @property
    def kernel(self):
        """Returns the current kernel on CPU with shape ``[1, kernel_len]``."""
        return self.kernel_cuda.detach().cpu()

    @property
    def input_size_reduced(self):
        """Returns the number of pixels reduced from the input image."""
        return self.kernel_cuda.shape[2] - 1

    def reset_parameters(self):
        """Resets the submodule :attr:`input_weight`."""
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5))

    def extra_repr(self):
        return '(input_weight): Parameter(size=%s)' \
            % str(tuple(self.input_weight.size()))

    def forward(self, x):
        return F.conv2d(x, self.kernel_cuda)


class KernelNet(_KernelNet):
    """The network outputs a 1D blur kernel to estimate slice selection.

    """
    def _calc_input_weight_size(self, ks):
        return Config().kernel_length + (ks - 1) * Config().kn_num_convs 

    def _create_conv(self, out_channels, kernel_size):
        in_channels = Config().kn_num_channels
        return nn.Conv2d(in_channels, out_channels, (kernel_size, 1))


class KernelNetZP(_KernelNet):
    """Zero-padded kernel net.

    """
    def _calc_input_weight_size(self, ks):
        return Config().kernel_length

    def _create_conv(self, out_channels, kernel_size):
        in_channels = Config().kn_num_channels
        padding = (kernel_size // 2, 0)
        return nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                         padding=padding)


class LowResDiscriminator(nn.Sequential):
    """Discriminator of low-resolution patches.

    """
    def __init__(self):
        super().__init__()
        config = Config()

        in_ch = 1
        for i, (ks, out_ch) in enumerate(zip(config.lrd_kernels[:-1],
                                             config.lrd_num_channels)):
            conv = nn.Conv2d(in_ch, out_ch, ks)
            conv = nn.utils.spectral_norm(conv)
            self.add_module('conv%d' % i, conv)
            relu = nn.LeakyReLU(config.lrelu_neg_slope)
            self.add_module('relu%d' % i, relu)
            in_ch = out_ch
        conv = nn.Conv2d(in_ch, 1, config.lrd_kernels[-1])
        conv = nn.utils.spectral_norm(conv)
        self.add_module('conv%d' % (i + 1), conv)
