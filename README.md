# Visualising and evaluating outputs of generative models

StyleGAN2, StyleGAN3, and StyleGAN-XL evaluation

Initially work by [Karras et al. 2021](https://github.com/NVlabs/stylegan3). But this page is mainly for reproducing work related to evaluation and visualisation.

## Requirements
- Linux.
- 1–8 high-end NVIDIA GPUs with at least 12 GB of memory.
- 64-bit Python 3.8 and PyTorch 1.9.0 (or later)
- CUDA toolkit 11.1 or later. 
- GCC 7 or later.
- Python libraries.

  ```conda env create -f environment.yml```
  
  ```conda activate stylegan3```
- FFHQ dataset (1028x1028)

### Evaluating
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python calc_metrics.py --metrics=fid50k_full --data=/data/ffhq/images1024x1024 --mirror=1 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl --gpus=4 
```
metrics:
- fid50k_full: Fréchet inception distance.
- kid50k_full: Kernel inception distance.
- pr50k3_full: Precision and recall.

data:
- path to dataset

pretrained network:
- StyleGAN2: https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl
- StyleGAN3: https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl 
- StyleGAN-XL: https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq1024.pkl

### Generating

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python gen_images.py --outdir=/data/nattharikasaetang/sg2output --trunc=1 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl --seeds=0-40
```

outdir:
- output images directory

seeds:
- amount of outputs

## License

Copyright © 2021, NVIDIA Corporation & affiliates. All rights reserved.

## Acknowledgments

Thank you authors of StyleGAN3 and StyleGAN-XL papers. This project can produce reliable results owing to them. Also, much thanks to Mac (my supervisor)
 and Justin who gave me a lot of useful knowledge and advice. 
