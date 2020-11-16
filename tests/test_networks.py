#!/usr/bin/env python

import torch
import numpy as np
from pathlib import Path
from pytorchviz import make_dot
from PIL import Image
import matplotlib.pyplot as plt

from spest.networks import KernelNet, LowResDiscriminator
from spest.config import Config


def test_networks():
    dirname = Path('results_networks')
    dirname.mkdir(exist_ok=True)
    image = np.array(Image.open('lena.png').convert('L'))
    image_cuda = torch.tensor(image).float().cuda()[None, None, ...]

    kn = KernelNet().cuda()
    print(kn)
    assert kn(image_cuda).shape == (1, 1, 492, 512)
    assert torch.sum(kn.kernel_cuda) == 1
    kn_dot = make_dot(image_cuda, kn)
    kn_dot.render(dirname.joinpath('kn'))

    lrd = LowResDiscriminator().cuda()
    print(lrd)
    assert lrd(image_cuda).shape == (1, 1, 502, 512)
    lrd_dot = make_dot(image_cuda, lrd)
    lrd_dot.render(dirname.joinpath('lrd'))


if __name__ == '__main__':
    test_networks()
