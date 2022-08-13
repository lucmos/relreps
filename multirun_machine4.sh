#!/bin/bash
# Reconstruction

# Absolute
python src/rae/run.py -m \
  core.tags='[reconstruction, absolute, tab1, resweep]' \
  'train.seed_index=0,1,2,3,4,5' \
  'nn/data/datasets=vision/fmnist,vision/cifar100,vision/mnist,vision/cifar10' \
  nn/module=autoencoder \
  nn/module/model=vae \
  train=reconstruction \
  nn.module.model.latent_dim=256 \
  "nn.module.model.hidden_dims=[32,64,128,256]" \
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=100 \
  nn.module.model.kld_weight=0.0002


# Classifcation
## Relative
#python src/rae/run.py -m \
#  core.tags='[classification, relative, tab1]' \
#  'nn/data/datasets=vision/fmnist,vision/cifar100,vision/mnist,vision/cifar10' \
#  nn/module=classifier \
#  nn/module/model=vision/relresnet \
#  nn.module.model.resnet_size=18 \
#  nn.module.model.input_size=224 \
#  nn.module.model.use_pretrained=True \
#  nn.module.model.finetune=False \
#  nn.module.model.hidden_features=512 \
#  nn.module.model.relative_attention.output_normalization_mode=layernorm \
#  'train.seed_index=0,1,2,3,4,5' \
#  train.trainer.max_epochs=10
#
## Absolute
#python src/rae/run.py -m \
#  core.tags='[classification, absolute, tab1]' \
#  'nn/data/datasets=vision/fmnist,vision/cifar100,vision/mnist,vision/cifar10' \
#  nn/module=classifier \
#  nn/module/model=vision/resnet \
#  nn.module.model.resnet_size=18 \
#  nn.module.model.input_size=224 \
#  nn.module.model.use_pretrained=True \
#  nn.module.model.finetune=False \
#  nn.module.model.hidden_features=512 \
#  'train.seed_index=0,1,2,3,4,5' \
#  train.trainer.max_epochs=10


