#!/usr/bin/env python

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from espreso2.networks import KernelNet
from espreso2.config import Config


cp_filename = '../results/simu-oasis3_lr-2e-4_bs-64_ne-20000_ie-200_sw-1_wd-1e-1_ps-16_lrdk-3,1-3,1-3,1-1,1-1,1_lrdc-64-64-64-64_knc-3_orth_clip/type-rect_fwhm-5p0_scale-0p25_len-13/checkpoint/epoch-15000.pt'

config_filename = '../results/simu-oasis3_lr-2e-4_bs-64_ne-20000_ie-200_sw-1_wd-1e-1_ps-16_lrdk-3,1-3,1-3,1-1,1-1,1_lrdc-64-64-64-64_knc-3_orth_clip/type-rect_fwhm-5p0_scale-0p25_len-13/config.json'

config = Config()
config.load_json(config_filename)

checkpoint = torch.load(cp_filename)
kn = KernelNet()
kn.load_state_dict(checkpoint['model_state_dict']['kernel_net'])

print(kn)

weight = kn.input_weight
print(weight.shape)
weight = weight.squeeze().cpu().detach().numpy()

dirname = Path('results_weights')
dirname.mkdir(exist_ok=True)

for i in range(weight.shape[0]):
    fig = plt.figure()
    plt.plot(weight[i, :])
    filename = dirname.joinpath('weight%d.png' % i)
    fig.savefig(filename)
    plt.close(fig)
