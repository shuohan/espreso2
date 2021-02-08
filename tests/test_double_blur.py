#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import convolve2d


output_dir = Path('results_double_blur')
output_dir.mkdir(exist_ok=True)

kernel_length = 41
shape = (128, 128)
radius = shape[0] / 4
grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
image = (grid[0] - shape[0]/2) ** 2 + (grid[1] - shape[1]/2) ** 2 < radius ** 2
# image = np.zeros(shape)
# image[shape[0]//4:-(shape[0]//4), shape[1]//4:-(shape[1]//4)] = 1

kernel = np.zeros(kernel_length)
kernel[kernel_length//4 : -(kernel_length//4)] = 1
kernel = kernel / np.sum(kernel)
kernel = kernel[:, None]

blur = convolve2d(image, kernel, mode='same')
x = int(shape[0] / 3)
# line_im 

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(image, interpolation='nearest', cmap='gray')
plt.plot([0, shape[1]], [x] * 2, '--', color='tab:orange')
plt.plot([x] * 2, [0, shape[0]], '-', color='tab:orange')
plt.xlim([0, shape[1]])
plt.ylim([0, shape[0]])
plt.axis(False)
plt.title('Image')

plt.subplot(2, 2, 2)
plt.imshow(blur, interpolation='nearest', cmap='gray')
plt.plot([0, shape[1]], [x] * 2, '--', color='tab:blue')
plt.plot([x] * 2, [0, shape[0]], '-', color='tab:blue')
plt.xlim([0, shape[1]])
plt.ylim([0, shape[0]])
plt.axis(False)
plt.title('Vertical Blur')

plt.subplot(2, 2, 3)
plt.plot(image[x, :], '--', color='tab:orange')
plt.plot(blur[x, :], '--', color='tab:blue')
plt.title('Horizontal curves')

plt.subplot(2, 2, 4)
plt.plot(image[:, x], '-', color='tab:orange')
plt.plot(blur[:, x], '-', color='tab:blue')
plt.title('Vertical curves')

fig.savefig(output_dir.joinpath('image.png'))
