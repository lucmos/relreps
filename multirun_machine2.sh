#!/bin/bash
# Reconstruction

# Absolute
python src/rae/run.py -m \
  core.tags='[reconstruction, absolute, tab1-reconstruction, dev]' \
  'train.seed_index=0' \
  'nn/data/datasets=vision/cifar10_nonorm' \
  nn/module=autoencoder \
  nn/module/model=ae \
  train=reconstruction \
  nn.module.model.latent_dim=256 \
  "nn.module.model.hidden_dims=null" \
  "nn.module.optimizer.lr=5e-4" \
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=60
