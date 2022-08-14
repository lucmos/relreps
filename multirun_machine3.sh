#!/bin/bash

# Absolute
python src/rae/run.py -m \
  core.tags='[reconstruction, absolute, tab1-definitive]' \
  'nn/data/datasets=vision/mnist,vision/fmnist,vision/cifar100_nonorm,vision/cifar10_nonorm' \
  'train.seed_index=0,1,2,3,4,5' \
  nn/module=autoencoder \
  nn/module/model=ae,vae \
  train=reconstruction \
  nn.module.model.latent_dim=500 \
  nn.data.anchors_num=500 \
  "nn.module.model.hidden_dims=null" \
  "nn.module.optimizer.lr=5e-4" \
  train.trainer.max_epochs=100
