# Latent Intuitive Physics
[Project Page](https://sites.google.com/view/latent-intuitive-physics/) | [Paper](https://openreview.net/pdf?id=WZu4gUGN13)



Code for our paper:
**Latent Intuitive Physics: Learning to Transfer Hidden Physics from a 3D Video**.

Xiangming Zhu, Huayu Deng, Haochen Yuan, [Yunbo Wang](https://wyb15.github.io/)<sup>†</sup>, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)

ICLR 2024

<img  src="/figure/lip.png"  alt="lip"  style="zoom:67%;"  />

## Dependencies
1. Create an environment
    ```bash
    conda create -n lip-env python=3.9
    conda activate lip-env
    ```
2. PyTorch and PyTorch3D

    Install PyTorch and PyTorch3D following the [official guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
    ```bash
    conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub
    conda install pytorch3d -c pytorch3d
    ```
4. Other dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Fetch data
Download the data and pretrained checkpoints from [Google Drive](https://drive.google.com/drive/folders/1ndvRbREQ4ahoqaoh9BpnJQWFCh9TnGS0?usp=drive_link) and place them under `./data` and `./pretrained_ckpts`.

## Run the training script
- Stage B: Visual posterior inference
```
python train_lip.py --expdir exps/cuboid_2000_0.065/stageB \
                            --expname latent1e-4 \
                            --config configs/stageB/cuboid.yaml \
                            --dataset cuboid_2000_0.065
```
- Stage C: Physical prior adaptation
```bash
python train_lip.py --expdir exps/cuboid_2000_0.065/stageC \
                            --expname encoder1e-4/ \
                            --config configs/stageC/cuboid.yaml \
                            --dataset cuboid_2000_0.065
```

## Citation

If you find our work helps, please cite our paper.

```bibtex
@inproceedings{zhu2024lip,
  author={Zhu, Xiangming and Deng, Huayu and Yuan, Haochen and Wang, Yunbo and Yang, Xiaokang},
  title={Latent Intuitive Physics: Learning to Transfer Hidden Physics from a 3D Video},
  booktitle = {International Conference on Learning Representations},
  year={2024}
}

```


##


## Acknowledgement
The implementation is based on the following repos:

https://github.com/syguan96/NeuroFluid

https://github.com/isl-org/DeepLagrangianFluids
