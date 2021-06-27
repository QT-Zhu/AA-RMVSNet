# AA-RMVSNet
## Description

Code for [AA-RMVSNet: Adaptive Aggregation Recurrent Multi-view Stereo Network]().

## Data Preparation



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



### Citation   
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

