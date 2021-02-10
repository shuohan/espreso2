# ESPRESO2: Estimate Slice Profile from a Single Image Only Version 2

| **[Docker Image][docker-image]** | **[Singularity Image][singularity-image]** | **[Documentation][docs]** |


## Introduction

This algorithm estimates a slice profile from a single 2D MR acquisition. 2D MR acquisitions usually have a lower through-plane resolution than in-plane resolutions. The rationale of ESPRESO2 is that if we use a correct slice profile to blur a high-resolution in-plane axis, its appearance should match the low-resolution through-plane axis. Therefore, we use a GAN to learn this slice profile by matching the distributions of image patches that are extracted along the in- and through-plane axes.

<img src="docs/source/_static/images/flowchart.svg" width="600"/>

<b>Figure 1</b>: Flowchart of ESPRESO2. <b>G</b>: the GAN's generator. <b>D</b>: the GAN's discriminator. <b>T</b>: transpose.

[docker-image]: link1
[singularity-image]: link2

## Installation

The [Docker image][docker-image] or [Singularity image][singularity-image] are recommended. The other option is to use `pip`:

```bash
pip install git+https://gitlab.com/shan-deep-networks/espreso2
```

## Usage

To use the Docker image, run

```bash
image=/path/to/image
output_dir=/path/to/output_dir
docker run -v $image:$image -v $output_dir:$output_dir --user $(id -u):$(id -g) \
    --rm --gpus device=0 -t espreso2 train.py -i $image -o $output_dir
```

To use the Singularity image, run
```bash
singularity run -B $image:$image -B $output_dir:$output_dir --nv \
    espreso2 train.py -i $image -o $output_dir
```

If `espreso2` is installed in the host machine, run

```bash
train.py -i $image -o $output_dir
```

[docker-image]: link1
[singularity-image]: link2
[docs]: link3
