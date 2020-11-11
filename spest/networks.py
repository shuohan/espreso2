import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from .config import Config


class KernelNet(nn.Sequential):
    """The network outputs a 1D blur kernel to estimate slice selection.

    """
    def __init__(self):
        super().__init__()
        config = Config()

        self.input_weight = nn.Parameter(torch.zeros(1, config.kn_num_channels))
        self.input_relu = nn.ReLU6()
        for i in range(config.kn_num_linears - 1):
            linear = nn.Linear(config.kn_num_channels, config.kn_num_channels)
            self.add_module('linear%d' % i, linear)
            self.add_module('relu%d' % i, nn.ReLU6())
        linear = nn.Linear(config.kn_num_channels, config.kernel_length)
        self.add_module('linear%d' % (config.kn_num_linears - 1), linear)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

        self._kernel_cuda = None
        self.register_buffer('avg_kernel', self._calc_kernel().detach())

    def reset_parameters(self):
        """Resets :attr:`input_weight`."""
        nn.init.kaiming_uniform_(self.input_weight, a=np.sqrt(5))

    def extra_repr(self):
        return '(input_weight): Parameter(size=%s)' \
            % str(tuple(self.input_weight.size()))

    def _calc_kernel(self):
        """Calculates the current kernel."""
        kernel = self.input_relu(self.input_weight)
        for module in self:
            kernel = module(kernel)
        return kernel

    def _reshape_kernel(self, kernel):
        """Reshapes the kernel to be compatible with the image shape."""
        raise NotImplementedError

    def update_kernel(self):
        """Updates the current kernel and calculates the moving average."""
        beta = Config().kernel_avg_beta
        self._kernel_cuda = self._calc_kernel()
        self.avg_kernel = (1 - beta) * self._kernel_cuda.detach() \
            + beta * self.avg_kernel

    @property
    def kernel_cuda(self):
        """Returns the current kernel on CUDA."""
        if self._kernel_cuda is None:
            self._kernel_cuda = self._calc_kernel()
        return self._kernel_cuda

    @property
    def kernel(self):
        """Returns the current kernel on CPU."""
        return self.kernel_cuda.detach().cpu()

    @property
    def input_size_reduced(self):
        """Returns the number of pixels reduced from the input image."""
        return Config().kernel_length - 1

    def forward(self, x):
        kernel = self.kernel_cuda.view(1, 1, -1, 1)
        return F.conv2d(x, kernel)


class LowResDiscriminator(nn.Sequential):
    """Discriminator to low-resolution patches.

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
        return nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True)
