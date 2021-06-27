# AA-RMVSNet
## Description

Code for AA-RMVSNet.

## Data Preparation

The preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) can be downloaded from the original [MVSNet repo](https://github.com/YoYo000/MVSNet).

## How to run
1. Install required dependencies:
   ```bash
   conda create -n drmvsnet python=3.6
   conda activate drmvsnet
   conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
   conda install -c conda-forge py-opencv plyfile tensorboardx
   ```

2. Set root of datasets as env variables in `env.sh`.
3. Train AA-RMVSNet on DTU dataset (note that training requires at least RTX TITAN GPUs):
   ```bash
   ./train_dtu.sh
   ```
4. Predict depth maps and fuse them to get point clouds (DTU dataset):
   ```bash
   ./eval_dtu.sh
   ./fusion.sh
   ```

## Citation   

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```

## Acknowledgements
This repository is heavily based on [Xiaoyang Guo](https://github.com/xy-guo/MVSNet_pytorch)'s PyTorch implementation.

