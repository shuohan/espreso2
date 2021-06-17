from pathlib import Path

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
                 weight_kernel_size, weight_stride):
        self.image = image
        self.patch_size = patch_size
        self.x = x
        self.y = y
        self.z = z
        self.voxel_size = voxel_size
        self.weight_kernel_size = weight_kernel_size
        self.weight_stride = weight_stride
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

    def save_figures(self, dirname):
        for orient, obj in self._figure_pool:
            obj.save_figures(Path(dirname, orient), d3=False)

    def _build_patches(self, orient):
        if orient == 'xz':
            return Patches(self.patch_size, self.image, x=self.x, y=self.y,
                           z=self.z, voxel_size=self.voxel_size).cuda()
        elif orient == 'yz':
            return Patches(self.patch_size, self.image, x=self.y, y=self.x,
                           z=self.z, voxel_size=self.voxel_size).cuda()

    def _build_trans_patches(self, patches):
        return [TransformedPatches(patches, f) for f in self._build_flips()]

    def _build_flips(self):
        """Flips x, z, or xz."""
        return Flip((0, )), Flip((1, )), Flip((0, 1))


class SamplerBuilderUniform(SamplerBuilder):
    """Builds a :class:`sssrlib.sample.Sampler` to uniformly sample patches.

    """
    def build(self):
        samplers = list()
        for orient in ['xz', 'yz']:
            patches = self._build_patches(orient)
            trans_patches = self._build_trans_patches(patches)
            samplers.extend([Sampler(p) for p in [patches] + trans_patches])
        self._sampler_xy = SamplerCollection(*samplers)
        self._sampler_z = self._sampler_xy
        return self

    def save_figures(self, dirname):
        pass


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
                                  stride=weight_self.stride)
        self._figure_pool.append((orient, agg))
        self._figure_pool.append((orient, weights))
        return weights


class CalcFG:
    def __init__(self, image):
        self.fg_mask = calc_foreground_mask(image)
    def save_figures(self, dirname, d3=True):
        save_fig(dirname, self.fg_mask, 'fg_mask', d3=d3)


class SamplerBuilderFG(SamplerBuilder):
    """Builds a :class:`sssrlib.sample.Sampler` to sample patches in foreground.

    """
    def build(self):
        samplers = list()
        agg_kernel = calc_avg_kernel(self.patch_size)
        for orient in ['xz', 'yz']:
            patches = self._build_patches(orient)
            trans_patches = self._build_trans_patches(patches)
            calc_fg = CalcFG(patches.image)
            agg = Aggregate(agg_kernel, (calc_fg.fg_mask, ))
            w = SampleWeights(patches, agg.agg_images)
            w = SuppressWeights(w, kernel_size=self.weight_kernel_size,
                                stride=self.weight_stride)
            self._figure_pool.append((orient, calc_fg))
            self._figure_pool.append((orient, patches))
            self._figure_pool.append((orient, agg))
            self._figure_pool.append((orient, w))
            for p in [patches] + trans_patches:
                samplers.append(Sampler(p, w.weights_flat, w.weights_mapping))
        self._sampler_xy = SamplerCollection(*samplers)
        self._sampler_z = self._sampler_xy
        return self
