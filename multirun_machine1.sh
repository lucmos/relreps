#!/bin/bash

# Absolute
python src/rae/run.py -m \
  core.tags='[reconstruction, absolute, fig:ae-rotations, less-small-ae]' \
  'nn/data/datasets=vision/mnist' \
  'train.seed_index=0,1,2,3,4,5,6,7,8,9' \
  nn/module=autoencoder \
  nn/module/model=ae \
  train=reconstruction \
  nn.module.model.latent_dim=2 \
  nn.data.anchors_num=500 \
  "nn.module.model.hidden_dims=[8, 16, 32, 64]" \
  "nn.module.optimizer.lr=1e-3" \
  train.trainer.max_epochs=20 \
  "+nn.module.model.latent_activation=null"
