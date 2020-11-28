import torch
import math
from torch import nn
import torch.nn.functional as F

from .config import Config


class KernelNet(nn.Sequential):
    """The network outputs a 1D blur kernel to estimate slice selection.

    """
    def __init__(self):
        super().__init__()
        config = Config()

        ks = 3
        num_ch = config.kn_num_channels

        weight_size = config.kernel_length + (ks - 1) * config.kn_num_convs 
        shape = (1, config.kn_num_channels, weight_size, 1)
        self.input_weight = nn.Parameter(torch.zeros(*shape))
        for i in range(config.kn_num_convs - 1):
            self.add_module('conv%d' % i, nn.Conv2d(num_ch, num_ch, (ks, 1)))
            self.add_module('relu%d' % i, nn.ReLU())
        conv = nn.Conv2d(num_ch, 1, (ks, 1))
        self.add_module('conv%d' % (config.kn_num_convs - 1), conv)
        self.softmax = nn.Softmax(dim=2)
        self.reset_parameters()

        # Calling self._calc_kernel() at init cannot put tensors into cuda
        self._kernel_cuda = None
        self.register_buffer('avg_kernel', self._calc_kernel().detach())

    def _calc_kernel(self):
        """Calculates the current kernel with shape ``[1, kernel_length]``."""
        kernel = self.input_weight
        for module in self:
            kernel = module(kernel)
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
        for i, ks in enumerate(config.lrd_kernels[:-1]):
            conv = nn.Conv2d(in_ch, out_ch, (ks, 1))
            conv = nn.utils.spectral_norm(conv)
            self.add_module('conv%d' % i, conv)
            relu = nn.LeakyReLU(config.lrelu_neg_slope)
            self.add_module('relu%d' % i, relu)
            in_ch = out_ch
            out_ch = in_ch * 2
        conv = nn.Conv2d(in_ch, 1, (config.lrd_kernels[-1], 1))
        conv = nn.utils.spectral_norm(conv)
        self.add_module('conv%d' % (i + 1), conv)
