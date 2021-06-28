# AA-RMVSNet
## Description

Code for AA-RMVSNet.
<img src="doc/architecture.png", width="800">

## Data Preparation
- Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) (also available at [Baiduyun](https://pan.baidu.com/s/1Wb9E6BWCJu4wZfwxm_t4TQ#list/path=%2F), PW: s2v2).
- For other datasets, please follow the practice in [Yao Yao's MVSNet repo](https://github.com/YoYo000/MVSNet).


## How to run
1. Install required dependencies:
   ```bash
   conda create -n drmvsnet python=3.6
   conda activate drmvsnet
   conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
   conda install -c conda-forge py-opencv plyfile tensorboardx
   ```
2. Set root of datasets as env variables in `env.sh`.
3. Train AA-RMVSNet on DTU dataset (note that training requires at least 24GB GPU memory):
   ```bash
   ./scripts/train_dtu.sh
   ```
4. Predict depth maps and fuse them to get point clouds of DTU:
   ```bash
   ./scripts/eval_dtu.sh
   ./scripts/fusion.sh
   ```
5. Check `scripts/fusion.sh` and switch to Tanks and Temples.
6. Predict depth maps and fuse them to get point clouds of Tanks and Temples:
   ```bash
   ./scripts/eval_tnt.sh
   ./scripts/fusion.sh
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

