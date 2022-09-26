#!/bin/bash
# Reconstruction

# Absolute
python src/rae/run.py -m \
  core.tags='[reconstruction, absolute, vae-qualitative]' \
  'train.seed_index=0,1' \
  'nn/data/datasets=vision/fmnist,vision/mnist,vision/cifar10' \
  nn=vision \
  train=reconstruction \
  nn/data=default \
  nn/module=autoencoder \
  nn/module/model=vae \
  nn.module.model.latent_dim=500 \
  "nn.module.model.hidden_dims=null" \
  "nn.module.model.calibrated_loss=False" \
  "nn.module.model.kld_weight=0.0025" \
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=150

