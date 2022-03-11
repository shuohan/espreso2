#!/usr/bin/env python

import pandas as pd


filename = 'compare_with_and_without_sp/dataframe.csv'
df = pd.read_csv(filename)

df_mean = df.groupby(['FWHM', 'W/ SP']).mean()
# print(df.set_index('subject')['ssim'] < 0.99)
df_mean['PSNR'] = df_mean['PSNR'].apply(lambda x: '%.2f' % x)
df_mean['SSIM'] = df_mean['SSIM'].apply(lambda x: '%.4f' % x)
print(df_mean)

filename = 'smore.tex'
df_mean.T.to_latex(filename)
