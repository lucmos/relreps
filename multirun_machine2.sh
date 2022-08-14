#!/bin/bash
# Reconstruction

# Absolute
python src/rae/run.py -m \
  core.tags='[reconstruction, relative, same-latent_dim2]' \
  'train.seed_index=0' \
  'nn/data/datasets=vision/cifar10_nonorm' \
  nn/module=autoencoder \
  nn/module/model=rel_ae,rel_vae \
  train=reconstruction \
  nn.module.model.latent_dim=500 \
  nn.data.anchors_num=500 \
  "nn.module.model.hidden_dims=null" \
  "nn.module.optimizer.lr=5e-4" \
  train.trainer.max_epochs=20 \
  'nn.module.model.relative_attention.relative_attentions.0.normalization_mode=l2' \
  'nn.module.model.relative_attention.relative_attentions.0.values_mode=similarities' \
  'nn.module.model.relative_attention.relative_attentions.0.values_self_attention_nhead=null'

#python src/rae/run.py -m \
#  core.tags='[reconstruction, absolute, same-latent_dim2]' \
#  'train.seed_index=0' \
#  'nn/data/datasets=vision/cifar10_nonorm' \
#  nn/module=autoencoder \
#  nn/module/model=ae,vae \
#  train=reconstruction \
#  nn.module.model.latent_dim=500 \
#  nn.data.anchors_num=500 \
#  "nn.module.model.hidden_dims=null" \
#  "nn.module.optimizer.lr=5e-4" \
#  train.trainer.max_epochs=20

