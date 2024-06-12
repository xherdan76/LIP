# stage B
python train_lip.py --expdir exps/cuboid_2000_0.065/stageB \
                            --expname latent1e-4 \
                            --config configs/stageB/cuboid.yaml \
                            --dataset cuboid_2000_0.065

# stage C
python train_lip.py --expdir exps/cuboid_2000_0.065/stageC \
                            --expname encoder1e-4/ \
                            --config configs/stageC/cuboid.yaml \
                            --dataset cuboid_2000_0.065
