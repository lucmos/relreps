#!/bin/bash

# Classification

# Relative
python src/rae/run.py -m \
  core.tags='[classification, relative, definitive]' \
  nn/module/model=vision/relresnet \
  'nn/data/datasets=vision/mnist,vision/fmnist,vision/cifar100_nonorm,vision/cifar10_nonorm' \
  'train.seed_index=0,1,2,3,4,5' \
  nn/module=classifier \
  train=classification \
  nn.module.model.resnet_size=18 \
  nn.module.model.input_size=224 \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=False \
  nn.module.model.hidden_features=500 \
  nn.data.anchors_num=500 \
  "nn.module.optimizer.lr=5e-4" \
  nn.module.model.relative_attention.output_normalization_mode=layernorm \
  train.trainer.max_epochs=100 \
  'nn.module.model.relative_attention.relative_attentions.0.normalization_mode=l2' \
  'nn.module.model.relative_attention.relative_attentions.0.values_mode=similarities'  \
  'nn.module.model.relative_attention.relative_attentions.0.values_self_attention_nhead=null'


# Absolute
python src/rae/run.py -m \
  core.tags='[classification, absolute, definitive]' \
  nn/module/model=vision/resnet \
  'nn/data/datasets=vision/mnist,vision/fmnist,vision/cifar100_nonorm,vision/cifar10_nonorm' \
  'train.seed_index=0,1,2,3,4,5' \
  nn/module=classifier \
  train=classification \
  nn.module.model.resnet_size=18 \
  nn.module.model.input_size=224 \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=False \
  nn.module.model.hidden_features=500 \
  "nn.module.optimizer.lr=5e-4" \
  train.trainer.max_epochs=75 \
  nn.module.model.remove_encoder_last_activation=True


# Reconstruction

# Relative
python src/rae/run.py -m \
  core.tags='[reconstruction, relative, definitive]' \
  nn/module/model=rel_vae,rel_ae \
  'nn/data/datasets=vision/mnist,vision/fmnist,vision/cifar100_nonorm,vision/cifar10_nonorm' \
  'train.seed_index=0,1,2,3,4,5' \
  nn/module=autoencoder \
  train=reconstruction \
  nn.module.model.latent_dim=500 \
  nn.data.anchors_num=500 \
  "nn.module.model.hidden_dims=null" \
  "nn.module.optimizer.lr=5e-4" \
  train.trainer.max_epochs=100 \
  'nn.module.model.relative_attention.relative_attentions.0.normalization_mode=l2' \
  'nn.module.model.relative_attention.relative_attentions.0.values_mode=similarities'  \
  'nn.module.model.relative_attention.relative_attentions.0.values_self_attention_nhead=null' \
  nn.module.model.remove_encoder_last_activation=True

# Absolute
python src/rae/run.py -m \
  core.tags='[reconstruction, absolute, definitive]' \
  nn/module/model=vae,ae \
  'nn/data/datasets=vision/mnist,vision/fmnist,vision/cifar100_nonorm,vision/cifar10_nonorm' \
  'train.seed_index=0,1,2,3,4,5' \
  nn/module=autoencoder \
  train=reconstruction \
  nn.module.model.latent_dim=500 \
  nn.data.anchors_num=500 \
  "nn.module.model.hidden_dims=null" \
  "nn.module.optimizer.lr=5e-4" \
  train.trainer.max_epochs=75 \
  nn.module.model.remove_encoder_last_activation=True
