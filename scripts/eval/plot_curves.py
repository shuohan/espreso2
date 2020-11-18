#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-files', nargs='+')
parser.add_argument('-n', '--names', nargs='+')
parser.add_argument('-m', '--markers', nargs='+', type=int)
parser.add_argument('-x', '--x-column-name')
parser.add_argument('-y', '--y-column-name')
parser.add_argument('-o', '--output-file')
parser.add_argument('-yl', '--ylim', type=float, nargs=2)
args = parser.parse_args()


import pandas as pd
import matplotlib.pyplot as plt
from plotting_utils import get_colors


colors = get_colors(10, 'tab10')
assert len(args.input_files) <= len(colors)

if args.names is not None:
    assert len(args.names) == len(args.input_files)

fig = plt.figure()
for fn, color in zip(args.input_files, colors):
    df = pd.read_csv(fn)[[args.x_column_name, args.y_column_name]]
    plt.plot(df[args.x_column_name], df[args.y_column_name], color=color)


if args.names is not None:
    plt.legend(args.names)

if args.markers is not None:
    y = plt.gca().get_ylim()[1]
    ys = [y] * len(args.markers)
    print(args.markers, ys)
    plt.stem(args.markers, ys, linefmt='k--', markerfmt=' ', basefmt=' ')

if args.ylim is not None:
    plt.ylim(args.ylim)

plt.grid(True)
fig.savefig(args.output_file)
