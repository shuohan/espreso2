from enum import Enum

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.transform import Flip
from sssrlib.sample import SuppressWeights, SampleWeights, Aggregate
from sssrlib.sample import Sampler, SamplerCollection, ImageGradients
from sssrlib.utils import calc_avg_kernel, calc_foreground_mask



class SamplerBuilder:
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

    @property
    def sampler_xy(self):
        return self._sampler_xy

    @property
    def sampler_z(self):
        return self._sampler_z

    def build(self):
        raise NotImplementedError

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
    def build(self):
        samplers = list()
        for orient in ['xz', 'yz']:
            patches = self._build_patches(orient)
            trans_patches = self._build_trans_patches(patches)
            samplers.extend([Sampler(p) for p in [patches] + trans_patches])
        self._sampler_xy = SamplerCollection(*samplers)
        self._sampler_z = self._sampler_xy
        return self


class SamplerBuilderGrad(SamplerBuilder):
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
            w0 = self._calc_sample_weights(patches, g.gradients[0], agg_kernel)
            w2 = self._calc_sample_weights(patches, g.gradients[2], agg_kernel)
            for p in [patches] + trans_patches:
                samp_xy.append(Sampler(p, w0.weights_flat, w0.weights_mapping))
                samp_z.append(Sampler(p, w2.weights_flat, w2.weights_mapping))
        self._sampler_xy = SamplerCollection(*samp_xy)
        self._sampler_z = SamplerCollection(*samp_z)
        return self

    def _calc_sample_weights(self, patches, grad, agg_kernel):
        agg = Aggregate(agg_kernel, (grad, ))
        weights = SampleWeights(patches, agg.agg_images)
        weights = SuppressWeights(weights, kernel_size=self.weight_kernel_size,
                                  stride=weight_self.stride)
        return weights


class SamplerBuilderFG(SamplerBuilder):
    def build(self):
        samplers = list()
        agg_kernel = calc_avg_kernel(self.patch_size)
        for orient in ['xz', 'yz']:
            patches = self._build_patches(orient)
            trans_patches = self._build_trans_patches(patches)
            fg_mask = calc_foreground_mask(patches.image)
            agg = Aggregate(agg_kernel, (fg_mask, ))
            w = SampleWeights(patches, agg.agg_images)
            w = SuppressWeights(w, kernel_size=self.weight_kernel_size,
                                stride=self.weight_stride)
            for p in [patches] + trans_patches:
                samplers.append(Sampler(p, w.weights_flat, w.weights_mapping))
        self._sampler_xy = SamplerCollection(*samplers)
        self._sampler_z = self._sampler_xy
        return self
