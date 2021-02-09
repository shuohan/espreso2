#!/usr/bin/env python

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


dirname = Path('results_intensity/oasis3_fwhm-5p0_scale-0p5_len-13_ns-1_flip_sw-1e-1_ie-200_wd-1e-3_in-1000_bs-64_lrs/kernel/')

# metrics = list()
# metrics2 = list()
# ratio = list()
# num = 5
# prev = None
# 
# for i, filename in enumerate(sorted(dirname.glob('avg_epoch*.npy'))):
#     kernel = np.load(filename)
#     if prev is None:
#         prev = kernel
#     # m = np.sum(np.abs((kernel - prev)) / kernel)
#     # m = np.sum(np.abs(kernel - prev))
#     prev = kernel
# 
#     center = np.sum(kernel * np.arange(len(kernel)))
#     m = np.sum((np.arange(len(kernel)) - center) ** 3 * kernel)
#     # m2 = np.sum(((np.arange(len(kernel)) - center) ** 2) * kernel) ** 0.5
#     # m1 = np.sqrt(np.sum(kernel ** 2))
#     # m = np.sum(kernel ** 2)
#     # metrics2.append(m2)
#     metrics.append(m)
#     # ratio.append((np.std(metrics[-num:]) / np.abs(np.mean(metrics[-num:]))))
#     ratio.append(np.std(metrics[-num:]))
#     if i > 100: 
#         break
# 

def func(numbers):
    x = np.arange(len(numbers))[:, None]
    x = np.concatenate([x, np.ones_like(x)], axis=1)
    numbers = np.array(numbers)[:, None]
    x_inv = np.linalg.pinv(x)
    ab = x_inv @ numbers
    return ab[0]


num = 1000
filename = 'results_intensity/oasis3_fwhm-5p0_scale-0p5_len-13_ns-1_flip_sw-1e-1_ie-200_wd-1e-3_in-1000_bs-64_lrs/loss.csv'
df = pd.read_csv(filename)
metrics = df['stop_metric_avg']
# ratio = metrics.rolling(4).apply(func).rolling(num).mean()
# ratio = metrics.rolling(num).std() / metrics.rolling(num).mean()
# ratio = metrics.rolling(num).std()
# ratio = metrics.ewm(alpha=0.005).mean()
# ratio = ratio.rolling(num).std() / ratio.rolling(num).mean()
ratio = metrics.rolling(num).std() / metrics.rolling(num).mean()

fig, ax1 = plt.subplots()
ax1.plot(metrics, color='tab:blue')
## ax1.plot(metrics2, color='tab:green')
ax2 = ax1.twinx()
ax2.plot(np.log10(np.abs(ratio)), color='tab:orange')
# ax1.plot(ratio, color='tab:orange')
plt.grid(True)
plt.show()
