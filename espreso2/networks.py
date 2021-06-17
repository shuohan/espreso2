import torch
import math
from torch import nn
import torch.nn.functional as F


class _SliceProfileNet(nn.Sequential):
    """Abstract class for slice profile networks.

    Note:
        Call :meth:`update_slice_profile` after each update of network
        parameters.

    Args:
        num_channels (int): The number of feature channels.
        kernel_size (int): The size of kernel in convolutions.
        num_convs (int): The number of convolutions.
        sp_length (int): The lenght of the slice profile vector.
        sp_avg_beta (float): The exponential average beta for the slice profile.
            See :meth:`update_slice_profile` for more details.
        symm_sp (bool): Enforce symmetry of the slice profile if ``True``.

    """
    def __init__(self, num_channels=256, kernel_size=3, num_convs=3,
                 sp_length=21, sp_avg_beta=0.99, symm_sp=False):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_convs = num_convs
        self.sp_length = sp_length
        self.sp_avg_beta = sp_avg_beta
        self.symm_sp = symm_sp

        size = self._calc_embedded_vector_size()
        ev = torch.zeros(1, self.num_channels, size, 1)
        self.embeded_vector = nn.Parameter(ev)
        self.reset_parameters()

        for i in range(self.num_convs - 1):
            self.add_module('conv%d' % i, self._create_conv(self.num_channels))
            self.add_module('relu%d' % i, nn.ReLU())
        self.add_module('conv%d' % (self.num_convs - 1), self._create_conv(1))

        # Calling self._calc_kernel() at init cannot put tensors into cuda
        self._sp = None
        self.register_buffer('avg_slice_profile', self._calc_sp().detach())

    def _calc_embedded_vector_size(self):
        raise NotImplementedError

    def _create_conv(self, out_channels):
        raise NotImplementedError

    def _calc_sp(self):
        """Calculates the current slice profile with shape ``[1, 1, length]``.

        """
        sp = self.embeded_vector
        for module in self:
            sp = module(sp)
        if self.symm_sp:
            sp = self._make_symmetric_sp(sp)
        sp = F.softmax(sp, dim=2)
        return sp

    def _make_symmetric_sp(self, sp):
        return 0.5 * (sp + torch.flip(sp, (2, )))

    def update_slice_profile(self):
        r"""Updates the current slice profile and calculates the moving average.

        This method updates the moving average :math:`k_t` as:

        .. math::

            \bar{k}_t = (1 - \beta) k_t + \beta \bar{k}_{t-1}

        Note:
            This function should be called at each iteration after
            backpropagation.

        """
        self._sp = self._calc_sp()
        beta = self.sp_avg_beta
        self.avg_slice_profile = (1 - beta) * self._sp.detach() \
            + beta * self.avg_slice_profile

    @property
    def slice_profile(self):
        """Returns the current slice profile with shape ``[1, 1, length]``.

        """
        if self._sp is None:
            self._sp = self._calc_sp()
        return self._sp

    @property
    def input_size_reduced(self):
        """Returns the number of pixels reduced from the input image."""
        return self.slice_profile.shape[2] - 1

    def reset_parameters(self):
        """Resets the submodule :attr:`embeded_vector`."""
        nn.init.kaiming_uniform_(self.embeded_vector, a=math.sqrt(5))

    def extra_repr(self):
        return '(embeded_vector): Parameter(size=%s)' \
            % str(tuple(self.embeded_vector.size()))

    def forward(self, x):
        """Convolves the slice profile with the input image."""
        return F.conv2d(x, self._sp)


class SliceProfileNet(_SliceProfileNet):
    """The network that outputs a slice profile.

    """
    def _calc_embedded_vector_size(self):
        return self.sp_length + (self.kernel_size - 1) * self.num_convs 

    def _create_conv(self, out_channels):
        return nn.Conv2d(self.num_channels, out_channels, (self.kernel_size, 1))


class SliceProfileNetZP(_SliceProfileNet):
    """Zero-padded slice profile net.

    """
    def _calc_embedded_vector_size(self, ks):
        return self.sp_length

    def _create_conv(self, out_channels):
        return nn.Conv2d(self.num_channels, out_channels, (self.kernel_size, 1),
                         padding=(self.kernel_size // 2, 0))


class Discriminator(nn.Sequential):
    """Discriminator of patches.

    Args:
        nums_channels (iterable[int]): The number of feature channels of each
            convolution.
        kernel_sizes (iterable[iterable[int]]): The kernel size of each
            convolution.
        lrelu_neg_slope (float): The negative slope for leaky ReLUs.

    """
    def __init__(self, nums_channels=(64, 64, 64, 64),
                 kernel_sizes=((3, 1), (3, 1), (3, 1), (1, 1), (1, 1)),
                 lrelu_neg_slope=0.1):
        super().__init__()
        self.nums_channels = nums_channels
        self.kernel_sizes = kernel_sizes
        assert len(self.nums_channels) == len(self.kernel_sizes) - 1
        self.lrelu_neg_slope = lrelu_neg_slope

        in_ch = 1
        zip_nums = zip(self.kernel_sizes[:-1], self.nums_channels)
        for i, (ks, out_ch) in enumerate(zip_nums):
            conv = nn.Conv2d(in_ch, out_ch, ks)
            conv = nn.utils.spectral_norm(conv)
            self.add_module('conv%d' % i, conv)
            relu = nn.LeakyReLU(self.lrelu_neg_slope)
            self.add_module('relu%d' % i, relu)
            in_ch = out_ch
        conv = nn.Conv2d(in_ch, 1, self.kernel_sizes[-1])
        conv = nn.utils.spectral_norm(conv)
        self.add_module('conv%d' % (i + 1), conv)
