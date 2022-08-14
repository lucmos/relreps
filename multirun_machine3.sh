#!/bin/bash

# Relative
python src/rae/run.py -m \
  core.tags='[reconstruction, relative, gaussian]' \
  'train.seed_index=0' \
  'nn/data/datasets=vision/fmnist' \
  nn/module=autoencoder \
  nn/module/model=rel_ae \
  train=reconstruction \
  nn.module.model.latent_dim=256 \
  "nn.module.model.hidden_dims=null" \
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=60 \
  'nn.module.model.relative_attention.relative_attentions.0.normalization_mode=l2' \
  'nn.module.model.relative_attention.relative_attentions.0.values_mode=similarities'  \
  'nn.module.model.relative_attention.relative_attentions.0.values_self_attention_nhead=null' \
  'nn.module.model.relative_attention.relative_attentions.0.anchors_sampling_mode=gaussian' \
  'nn.module.model.relative_attention.relative_attentions.0.n_anchors_sampling_per_class=50'
