import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.ndimage import label as find_cc
from improc3d import calc_bbox3d, padcrop3d

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.transform import Flip
from sssrlib.sample import SuppressWeights, SampleWeights, Aggregate
from sssrlib.sample import Sampler, SamplerCollection, ImageGradients
from sssrlib.utils import calc_avg_kernel, calc_foreground_mask, save_fig



class SamplerBuilder:
    """Builds a :class:`sssrlib.sample.Sampler` instance.

    Example:
        >>> builder = SamplerBuilder(...).build()
        >>> sampler_xy = builder.sampler_xy
        >>> sampler_z = builder.sampler_z

    Args:
        patch_size (iterable[int]): The patch size to extract.
        x (int): The index of the x-axis.
        y (int): The index of the y-axis.
        z (int): The index of the z-axis.
        voxel_size (iterable[float]): The voxel size of the image.
        weight_kernel_size (iterable[int]): The kernel size to suppress weights
            using max-pooling.
        weight_stride (iterable[int]): The stride to suppress weights
            using max-pooling.

    """
    def __init__(self, patch_size, image, x, y, z, voxel_size,
                 weight_kernel_size, weight_stride, aug=False):
        self.image = image
        self.patch_size = patch_size
        self.x = x
        self.y = y
        self.z = z
        self.voxel_size = voxel_size
        self.weight_kernel_size = weight_kernel_size
        self.weight_stride = weight_stride
        self.aug = aug
        self._sampler_xy = None
        self._sampler_z = None
        self._figure_pool = list()

    @property
    def sampler_xy(self):
        """Returns the sampler to extract patches with high-res along axis 0."""
        return self._sampler_xy

    @property
    def sampler_z(self):
        """Returns the sampler to extract patches with low-res along axis 0."""
        return self._sampler_z

    def build(self):
        """Builds the sampler.

        Returns:
            self.

        """
        raise NotImplementedError

    def save_figures(self, dirname, d3=False):
        for orient, obj in self._figure_pool:
            obj.save_figures(Path(dirname, orient), d3=d3)

    def _get_orients(self):
        return ['xz', 'xz_f0', 'xz_f2', 'xz_f02',
                'yz', 'yz_f0', 'yz_f2', 'yz_f02']

    def _build_patches(self, orient):
        tmp = orient.split('_')
        orient = tmp[0]
        flip_axes = [int(f) for f in tmp[1][1:]] if len(tmp) > 1 else None
        if orient == 'xz':
            patches = Patches(self.patch_size, self.image, x=self.x, y=self.y,
                              z=self.z, voxel_size=self.voxel_size).cuda()
        elif orient == 'yz':
            patches = Patches(self.patch_size, self.image, x=self.y, y=self.x,
                              z=self.z, voxel_size=self.voxel_size).cuda()
        if flip_axes:
            patches.image = torch.flip(patches.image, flip_axes)
        return patches


class SamplerBuilderUniform(SamplerBuilder):
    """Builds a :class:`sssrlib.sample.Sampler` to uniformly sample patches.

    """
    def build(self):
        samplers = list()
        for orient in self._get_orients():
            patches = self._build_patches(orient)
            samplers.append(Sampler(patches))
        self._sampler_xy = SamplerCollection(*samplers)
        self._sampler_z = self._sampler_xy
        return self


class SamplerBuilderGrad(SamplerBuilder):
    """Builds a :class:`sssrlib.sample.Sampler` to sample patches according to
    image graidnets.

    """
    @property
    def sampler_xy(self):
        return self._sampler_xy

    @property
    def sampler_z(self):
        return self._sampler_z

    def build(self):
        samp_xy = list()
        samp_z = list()
        agg_kernel = calc_avg_kernel(self.patch_size)
        for orient in ['xz', 'yz']:
            patches = self._build_patches(orient)
            trans_patches = self._build_trans_patches(patches)
            g = ImageGradients(patches, sigma=1)
            self._figure_pool.append((orient, patches))
            self._figure_pool.append((orient, g))
            w0 = self._calc_weights(patches, g.gradients[0], agg_kernel, orient)
            w2 = self._calc_weights(patches, g.gradients[2], agg_kernel, orient)
            for p in [patches] + trans_patches:
                samp_xy.append(Sampler(p, w0.weights_flat, w0.weights_mapping))
                samp_z.append(Sampler(p, w2.weights_flat, w2.weights_mapping))
        self._sampler_xy = SamplerCollection(*samp_xy)
        self._sampler_z = SamplerCollection(*samp_z)
        return self

    def _calc_weights(self, patches, grad, agg_kernel, orient):
        agg = Aggregate(agg_kernel, (grad, ))
        weights = SampleWeights(patches, agg.agg_images)
        weights = SuppressWeights(weights, kernel_size=self.weight_kernel_size,
                                  stride=self.weight_stride)
        self._figure_pool.append((orient, agg))
        self._figure_pool.append((orient, weights))
        return weights


class CalcHeadMask:
    """Calculates a head mask.

    This class first calculates a foreground mask using Ostu's threshold (three
    classes case). The foreground mask is then closed and hole-filled. Finally,
    the largest connected component is extracted from the filled mask.

    Example:
        >>> CalcHeadMask(patches).cc

    Args:
        patches (sssrlib.patches.Patches): Holds the image to calculate the head
            mask from.

    """
    def __init__(self, patches):
        self.patches = patches
        self._fg_mask = calc_foreground_mask(self.patches.image)
        self._cc = self._calc_largest_cc()

    @property
    def fg_mask(self):
        """Returns the foreground mask calculated from Otsu's threshold."""
        return self._fg_mask

    @property
    def cc(self):
        """Returns the largest connected component."""
        return self._cc.to(self.patches.image)

    def _calc_largest_cc(self):
        pad = 10
        mask = self._fg_mask.cpu().numpy()
        padded = np.pad(mask, pad)
        filled = binary_closing(padded, iterations=pad)
        filled = binary_fill_holes(filled)
        filled = filled[pad:-pad, pad:-pad, pad:-pad]
        assert filled.shape == mask.shape
        labels, num_labels = find_cc(filled)
        counts = np.bincount(labels.flatten())
        for l in np.argsort(counts)[::-1]:
            cc = labels == l
            if filled[cc][0]:
                break
        cc = torch.Tensor(cc)
        return cc

    def save_figures(self, dirname, d3=True):
        save_fig(dirname, self._fg_mask, 'fg_mask', d3=d3)
        save_fig(dirname, self._cc, 'cc', d3=d3)


class CalcHeadMaskSimple:
    """Calculates a head mask using Ostu's thresholds (three classes).

    """
    def __init__(self, patches):
        self.patches = patches
        self._fg_mask = calc_foreground_mask(self.patches.image)

    @property
    def fg_mask(self):
        """Returns the foreground mask."""
        return self._fg_mask

    def save_figures(self, dirname, d3=True):
        save_fig(dirname, self._fg_mask, 'fg_mask', d3=d3)


class SamplerBuilderFG(SamplerBuilder):
    """Builds a :class:`sssrlib.sample.Sampler` to sample patches in foreground.

    :class:`sssrlib.sample.SuppressWeights` is used to reduced the number of
    possible patches. The foreground mask is closed and hole-filled.

    """
    def build(self):
        samplers = list()
        agg_kernel = calc_avg_kernel(self.patch_size)
        for orient in self._get_orients():
            patches = self._build_patches(orient)
            w = self._calc_weights(patches, agg_kernel, orient)
            samplers.append(Sampler(patches, w.weights_flat, w.weights_mapping))
        self._sampler_xy = SamplerCollection(*samplers)
        self._sampler_z = self._sampler_xy
        return self

    def _calc_weights(self, patches, agg_kernel, orient):
        calc_mask = CalcHeadMask(patches)
        weights = SampleWeights(patches, (calc_mask.cc, ))
        weights = SuppressWeights(weights, kernel_size=self.weight_kernel_size,
                                  stride=self.weight_stride)
        self._figure_pool.append((orient, calc_mask))
        self._figure_pool.append((orient, patches))
        self._figure_pool.append((orient, weights))
        return weights


class SamplerBuilderSimpleFG(SamplerBuilder):
    """Builds a :class:`sssrlib.sample.Sampler` to sample patches in foreground.

    """
    def build(self):
        samplers = list()
        agg_kernel = calc_avg_kernel(self.patch_size)
        for orient in self._get_orients():
            patches = self._build_patches(orient)
            w = self._calc_weights(patches, agg_kernel, orient)
            samplers.append(Sampler(patches, w.weights_flat))
        self._sampler_xy = SamplerCollection(*samplers)
        self._sampler_z = self._sampler_xy
        return self

    def _calc_weights(self, patches, agg_kernel, orient):
        calc_mask = CalcHeadMaskSimple(patches)
        weights = SampleWeights(patches, (calc_mask.fg_mask, ))
        self._figure_pool.append((orient, calc_mask))
        self._figure_pool.append((orient, patches))
        self._figure_pool.append((orient, weights))
        return weights


class SamplerBuilderAggFG(SamplerBuilder):
    """Builds a :class:`sssrlib.sample.Sampler` to sample patches in foreground.

    """
    def build(self):
        samplers = list()
        agg_kernel = calc_avg_kernel(self.patch_size)
        for orient in self._get_orients():
            patches = self._build_patches(orient)
            w = self._calc_weights(patches, agg_kernel, orient)
            samplers.append(Sampler(patches, w.weights_flat))
        self._sampler_xy = SamplerCollection(*samplers)
        self._sampler_z = self._sampler_xy
        return self

    def _calc_weights(self, patches, agg_kernel, orient):
        calc_mask = CalcHeadMaskSimple(patches)
        agg = Aggregate(agg_kernel, (calc_mask.fg_mask, ))
        weights = SampleWeights(patches, agg.agg_images)
        weights = SuppressWeights(weights, kernel_size=self.weight_kernel_size,
                                  stride=self.weight_stride)
        self._figure_pool.append((orient, calc_mask))
        self._figure_pool.append((orient, agg))
        self._figure_pool.append((orient, patches))
        self._figure_pool.append((orient, weights))
        return weights
