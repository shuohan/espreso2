from enum import Enum

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.transform import Flip
from sssrlib.sample import SuppressWeights, SampleWeights, Aggregate
from sssrlib.sample import Sampler, SamplerCollection, ImageGradients
from sssrlib.utils import calc_avg_kernel, calc_foreground_mask


class SamplerType(str, Enum):
    UNIFORM = 'uniform'
    GRADIENT = 'gradient'
    FOREGROUND = 'foreground'


class SamplerBuilder:
    pass


class SamplerBuilderUniform(SamplerBuilder):
    def __init__(self, patch_size, image, x, y, z, voxel_size):
        self.image = image
        self.patch_size = patch_size
        self.x = x
        self.y = y
        self.z = z
        self.voxel_size = voxel_size
        self._sampler = None

    @property
    def sampler(self):
        return self._sampler

    def build(self):
        samplers = list()
        for orient in ['xz', 'yz']:
            patches = self._build_patches(orient)
            trans_patches = self._build_trans_patches(patches)
            samplers.append(Sampler(patches))
            samplers.extend([Sampler(p) for p in trans_patches])
        self._sampler = SamplerCollection(*samplers)
        return self

    def _build_patches(self, orient):
        if orient == 'xz':
            patches = Patches(self.patch_size, self.image, x=self.x, y=self.y,
                              z=self.z, voxel_size=self.voxel_size)
        elif orient == 'yz':
            patches = Patches(self.patch_size, self.image, x=self.y, y=self.x,
                              z=self.z, voxel_size=self.voxel_size)
        else:
            assert False
        return patches.cuda()

    def _build_trans_patches(self, patches):
        return [TransformedPatches(patches, flip)
                for flip in self._build_flips()]

    def _build_flips(self):
        """Flips x, z, or xz."""
        return Flip((0, )), Flip((1, )), Flip((0, 1))


class SamplerBuilderGrad(SamplerBuilderUniform):
    def __init__(self, patch_size, image, x, y, z, voxel_size,
                 kernel_size, stride):
        super().__init__(patch_size, image, x, y, z, voxel_size)
        self.kernel_size = kernel_size
        self.stride = stride

    @property
    def sampler_xy(self):
        return self._sampler_xy

    @property
    def sampler_z(self):
        return self._sampler_z

    def build(self):
        samplers_xy = list()
        samplers_z = list()
        agg_kernel = calc_avg_kernel(self.patch_size)
        for orient in ['xz', 'yz']:
            patches = self._build_patches(orient)
            grads = ImageGradients(patches, sigma=1)
            agg0 = Aggregate(agg_kernel, (grads.gradients[0], ))
            agg2 = Aggregate(agg_kernel, (grads.gradients[2], ))

            weights0 = SampleWeights(patches, agg0.agg_images)
            weights0 = SuppressWeights(weights0, kernel_size=self.kernel_size, stride=self.stride)
            weights2 = SampleWeights(patches, agg2.agg_images)
            weights2 = SuppressWeights(weights2, kernel_size=self.kernel_size, stride=self.stride)

            trans_patches = self._build_trans_patches(patches)
            for p in [patches] + trans_patches:
                sampler0 = Sampler(p, weights0.weights_flat, weights0.weights_mapping)
                sampler2 = Sampler(p, weights2.weights_flat, weights2.weights_mapping)
                samplers_xy.append(sampler0)
                samplers_z.append(sampler2)

        self._sampler_xy = SamplerCollection(*samplers_xy)
        self._sampler_z = SamplerCollection(*samplers_xy)

        return self


class SamplerBuilderFG(SamplerBuilderUniform):

    def build(self):
        samplers = list()
        agg_kernel = calc_avg_kernel(self.patch_size)
        for orient in ['xz', 'yz']:
            patches = self._build_patches(orient)
            fg_mask = calc_foreground_mask(patches.image)
            agg = Aggregate(agg_kernel, (fg_mask, ))
            weights = SampleWeights(patches, agg.agg_images)
            weights = SuppressWeights(weights, kernel_size=self.patch_size,
                                      stride=self.patch_size)

            trans_patches = self._build_trans_patches(patches)
            for p in [patches] + trans_patches:
                sampler = Sampler(p, weights.weights_flat, weights.weights_mapping)
                samplers.append(sampler)
        self._sampler = SamplerCollection(*samplers)

        return self
