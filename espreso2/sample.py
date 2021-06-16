from enum import Enum

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.transform import Flip
from sssrlib.sample import Sampler, SamplerCollection


class SamplerType(str, Enum):
    UNIFORM = 'uniform'
    GRADIENT = 'gradient'


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
        for orient in ['xy', 'yx']:
            patches = self._build_patches(orient)
            trans_patches = self._build_trans_patches(patches)
            samplers.append(Sampler(patches))
            samplers.extend([Sampler(p) for p in trans_patches])
        self._sampler = SamplerCollection(*samplers)
        return self

    def _build_patches(self, orient):
        if orient == 'xy':
            patches = Patches(self.patch_size, self.image, x=self.x, y=self.y,
                              z=self.z, voxel_size=self.voxel_size)
        elif orient == 'yx':
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


class SamplerBuilderXYGrad:
    pass


class SamplerBuilderZGrad:
    pass
