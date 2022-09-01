#!/bin/bash
# Reconstruction

python src/rae/run.py -m \
  core.tags='[classification, absolute, fig:ae-rotations, small_cnn]' \
  'nn/data/datasets=vision/mnist' \
  'train.seed_index=0,1,2,3,4,5' \
  nn/module=classifier \
  nn/module/model=cnn \
  train=classification \
  nn.module.model.latent_dim=2 \
  nn.data.anchors_num=500 \
  "nn.module.model.hidden_dims=[2, 3, 4, 8]" \
  "nn.module.optimizer.lr=5e-4" \
  train.trainer.max_epochs=40
