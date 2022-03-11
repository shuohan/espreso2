#!/usr/bin/env python


import pandas as pd

# for ne in [10000, 11000, 12000, 13000, 14000, 15000]:
for ne in [15000]:

    filename = 'errors_epoch-{}/dataframe.csv'.format(ne)
    df = pd.read_csv(filename)

    df['scale'] = (1 / df['scale']).astype(int)
    df['fwhm'] = df['fwhm'].astype(int)

    df_mean = df.groupby(['type', 'fwhm', 'scale']).mean()
    # print(df.set_index('subject')['ssim'] < 0.99)

    print(df_mean.columns.values)

    df_mean['fwhm error'] = df_mean['fwhm error'].apply(lambda x: '%.2f' % x)
    df_mean['profile error'] = df_mean['profile error'].apply(lambda x: '%.2f' % x)
    df_mean['psnr'] = df_mean['psnr'].apply(lambda x: '%.2f' % x)
    df_mean['ssim'] = df_mean['ssim'].apply(lambda x: '%.4f' % x)


    df_mean = df_mean.rename(index={'gauss': 'Gaussian profile',
                                    'rect': 'Rect profile'},
                             columns={'fwhm error': 'F. Err.', 'profile error':
                                      'P. Err.', 'psnr': 'PSNR', 'ssim':
                                      'SSIM'}) 

    df_mean.index.names = ['', 'FWHM', 'Scale']

    df_mean_g = df_mean.loc['Gaussian profile']
    df_mean_r = df_mean.loc['Rect profile']

    df_mean_g = df_mean_g.T
    df_mean_r = df_mean_r.T

    # df_std = df.groupby(['type', 'fwhm', 'scale']).std()
    # df_std['fwhm error'] = df_std['fwhm error'].round(2)
    # df_std['profile error'] = df_std['profile error'].round(2)
    # df_std['psnr'] = df_std['psnr'].round(2).astype(str)
    # df_std['ssim'] = df_std['ssim'].round(4).astype(str)

    # print(ne, df_mean['ssim'].mean(), df_mean['ssim'].std())

    # print(df_mean)

    filename = 'errors_g.tex'
    df_mean_g.to_latex(filename)
    filename = 'errors_r.tex'
    df_mean_r.to_latex(filename)
    # print(df_std)
