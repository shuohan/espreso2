import torch
import math
from torch import nn
import torch.nn.functional as F

from .config import Config


class Clamp(nn.Module):
    """Wrapper class of :func:`torch.clamp`.

    """
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

    def extra_repr(self):
        return 'min={}, max={}'.format(self.min, self.max)


class Interpolate(nn.Module):
    """Wrapper class of :func:`torch.nn.functional.interpolate`.

    """
    def __init__(self, size=None, scale_factor=None, mode='bicubic',
                 align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corners,
                             recompute_scale_factor=self.recompute_scale_factor)

    def extra_repr(self):
        attrs = ['size', 'scale_factor', 'mode', 'align_corners',
                 'recompute_scale_factor']
        message = []
        for attr in attrs:
            value = getattr(self, attr)
            if value is not None:
                message.append('{}={}'.format(attr, value))
        message = ', '.join(message)
        return message


class KernelNet(nn.Sequential):
    """The network outputs a 1D blur kernel to estimate slice selection.

    """
    def __init__(self):
        super().__init__()
        config = Config()

        self.input_weight = nn.Parameter(torch.zeros(1, config.kn_num_channels))
        self.input_clamp = Clamp(-3, 3)
        for i in range(config.kn_num_linears - 1):
            linear = nn.Linear(config.kn_num_channels, config.kn_num_channels)
            self.add_module('linear%d' % i, linear)
            self.add_module('clamp%d' % i, Clamp(-3, 3))
        linear = nn.Linear(config.kn_num_channels, config.kernel_length)
        self.add_module('linear%d' % (config.kn_num_linears - 1), linear)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

        # Calling self._calc_kernel() at init cannot put tensors into cuda
        self._kernel_cuda = None
        self.register_buffer('avg_kernel', self._calc_kernel().detach())

    def _calc_kernel(self):
        """Calculates the current kernel with shape ``[1, kernel_length]``."""
        kernel = self.input_clamp(self.input_weight)
        for module in self:
            kernel = module(kernel)
        kernel = kernel.view(1, 1, -1, 1)
        return kernel

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


class LowResDiscriminator(nn.Sequential):
    """Discriminator of low-resolution patches.

    """
    def __init__(self):
        super().__init__()
        config = Config()

        in_ch = 1
        out_ch = config.lrd_num_channels
        for i in range(config.lrd_num_convs - 1):
            conv = self._create_conv(in_ch, out_ch)
            conv = nn.utils.spectral_norm(conv)
            self.add_module('conv%d' % i, conv)
            relu = nn.LeakyReLU(config.lrelu_neg_slope)
            self.add_module('relu%d' % i, relu)
            in_ch = out_ch
            out_ch = in_ch * 2

        conv = self._create_conv(in_ch, 1)
        conv = nn.utils.spectral_norm(conv)
        self.add_module('conv%d' % (config.lrd_num_convs - 1), conv)

    def _create_conv(self, in_ch, out_ch):
        """Creates a conv layer."""
        return nn.Conv2d(in_ch, out_ch, Config().lrd_kernel_size, padding=0)
