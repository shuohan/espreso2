#!/usr/bin/env bash

scp mashie:/iacl/pg20/shuo/work/psf/data/flair/20121_02_FLAIRPre_2D.nii ./
scp mashie:/iacl/pg20/shuo/work/psf/data/t1/20208_03_T1Post.nii ./

scp mashie:/iacl/pg20/shuo/work/psf/results_kn-11_sw-1.0/t1/20208_03_T1Post/kernel/avg_epoch-30000.npy 20208_03_T1Post_kernel.npy
scp mashie:/iacl/pg20/shuo/work/psf/results_kn-11_sw-1.0/flair/20121_02_FLAIRPre_2D/kernel/avg_epoch-30000.npy 20121_02_FLAIRPre_2D_kernel.npy
