#!/bin/bash
# Reconstruction

# Relative
python src/rae/run.py -m \
  core.tags='[reconstruction, relative, tab1-reconstruction]' \
  'train.seed_index=0,1,2,3,4,5' \
  'nn/data/datasets=vision/fmnist,vision/cifar100,vision/mnist,vision/cifar10' \
  nn/module=autoencoder \
  nn/module/model=rel_vae,rel_ae \
  train=reconstruction \
  nn.module.model.latent_dim=256 \
  "nn.module.model.hidden_dims=null" \
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=60 \
  'nn.module.model.relative_attention.relative_attentions.0.normalization_mode=l2' \
  'nn.module.model.relative_attention.relative_attentions.0.values_mode=similarities'  \
  'nn.module.model.relative_attention.relative_attentions.0.values_self_attention_nhead=null'


# Absolute
python src/rae/run.py -m \
  core.tags='[reconstruction, absolute, tab1-reconstruction]' \
  'train.seed_index=0,1,2,3,4,5' \
  'nn/data/datasets=vision/fmnist,vision/cifar100,vision/mnist,vision/cifar10' \
  nn/module=autoencoder \
  nn/module/model=vae,ae \
  train=reconstruction \
  nn.module.model.latent_dim=256 \
  "nn.module.model.hidden_dims=null" \
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=60
