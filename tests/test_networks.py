#!/usr/bin/env python

import torch
import numpy as np
from pathlib import Path
from pytorchviz import make_dot
from PIL import Image
import matplotlib.pyplot as plt

from espreso2.networks import KernelNet, LowResDiscriminator, KernelNetZP
from espreso2.config import Config


def test_networks():
    dirname = Path('results_networks')
    dirname.mkdir(exist_ok=True)
    image = np.array(Image.open('lena.png').convert('L'))
    image_cuda = torch.tensor(image).float().cuda()[None, None, ...]

    config = Config()
    config.kn_kernel_size = 2
    config.kn_num_channels = 256
    config.kn_num_convs = 3

    kn = KernelNet().cuda()
    print(kn)
    assert kn.kernel_cuda.shape == (1, 1, 19, 1)
    assert kn(image_cuda).shape == (1, 1, 494, 512)
    assert torch.isclose(torch.sum(kn.kernel_cuda), torch.tensor(1).float())
    kn_dot = make_dot(image_cuda, kn)
    kn_dot.render(dirname.joinpath('kn'))

    fig = plt.figure()
    kernel = kn.kernel.numpy().squeeze()
    plt.plot(kernel)
    fig.savefig(dirname.joinpath('kernel.png'))

    lrd = LowResDiscriminator().cuda()
    print(lrd)
    assert lrd(image_cuda).shape == (1, 1, 502, 502)
    lrd_dot = make_dot(image_cuda, lrd)
    lrd_dot.render(dirname.joinpath('lrd'))
# 
#     kn = KernelNetZP().cuda()
#     print(kn)
#     assert kn.kernel_cuda.shape == (1, 1, 19, 1)
#     assert kn(image_cuda).shape == (1, 1, 494, 512)
#     assert torch.isclose(torch.sum(kn.kernel_cuda), torch.tensor(1).float())
#     kn_dot = make_dot(image_cuda, kn)
#     kn_dot.render(dirname.joinpath('knzp'))
# 
#     fig = plt.figure()
#     kernel = kn.kernel.numpy().squeeze()
#     plt.plot(kernel)
#     fig.savefig(dirname.joinpath('kernel_zp.png'))


if __name__ == '__main__':
    test_networks()
