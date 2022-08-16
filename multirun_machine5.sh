#!/bin/bash
# Reconstruction

# Relative
python src/rae/run.py -m \
  core.tags='[reconstruction, relative, tanh-encoder-decoder]' \
  'nn/data/datasets=vision/mnist' \
  'train.seed_index=0,1,2,3,4,5' \
  nn/module=autoencoder \
  nn/module/model=rel_ae \
  train=reconstruction \
  nn.module.model.latent_dim=500 \
  nn.data.anchors_num=500 \
  "nn.module.model.hidden_dims=null" \
  "nn.module.optimizer.lr=5e-4" \
  train.trainer.max_epochs=40 \
  'nn.module.model.relative_attention.relative_attentions.0.normalization_mode=l2' \
  'nn.module.model.relative_attention.relative_attentions.0.values_mode=similarities'  \
  'nn.module.model.relative_attention.relative_attentions.0.values_self_attention_nhead=null' \
  nn.module.model.activation=torch.nn.Tanh


## Absolute
#python src/rae/run.py -m \
#  core.tags='[reconstruction, absolute, tab1-definitive]' \
#  'nn/data/datasets=vision/mnist,vision/fmnist,vision/cifar100_nonorm,vision/cifar10_nonorm' \
#  'train.seed_index=0,1,2,3,4,5' \
#  nn/module=autoencoder \
#  nn/module/model=ae,vae \
#  train=reconstruction \
#  nn.module.model.latent_dim=500 \
#  nn.data.anchors_num=500 \
#  "nn.module.model.hidden_dims=null" \
#  "nn.module.optimizer.lr=5e-4" \
#  train.trainer.max_epochs=100

## Relative
#python src/rae/run.py -m \
#  core.tags='[reconstruction, relative, tab1-definitive]' \
#  'nn/data/datasets=vision/mnist,vision/fmnist,vision/cifar100_nonorm,vision/cifar10_nonorm' \
#  'train.seed_index=0,1,2,3,4,5' \
#  nn/module=autoencoder \
#  nn/module/model=rel_vae,rel_ae \
#  train=reconstruction \
#  nn.module.model.latent_dim=500 \
#  nn.data.anchors_num=500 \
#  "nn.module.model.hidden_dims=null" \
#  "nn.module.optimizer.lr=5e-4" \
#  train.trainer.max_epochs=100 \
#  'nn.module.model.relative_attention.relative_attentions.0.normalization_mode=l2' \
#  'nn.module.model.relative_attention.relative_attentions.0.values_mode=similarities'  \
#  'nn.module.model.relative_attention.relative_attentions.0.values_self_attention_nhead=null'

## Relative Quantization
#python src/rae/run.py -m \
#  core.tags='[reconstruction, relative, tab1-definitive, quantized]' \
#  'nn/data/datasets=vision/mnist,vision/fmnist,vision/cifar100_nonorm,vision/cifar10_nonorm' \
#  'train.seed_index=0,1,2,3,4,5' \
#  nn/module=autoencoder \
#  nn/module/model=rel_vae,rel_ae \
#  train=reconstruction \
#  nn.module.model.latent_dim=500 \
#  nn.data.anchors_num=500 \
#  "nn.module.model.hidden_dims=null" \
#  "nn.module.optimizer.lr=5e-4" \
#  train.trainer.max_epochs=100 \
#  'nn.module.model.relative_attention.relative_attentions.0.normalization_mode=l2' \
#  'nn.module.model.relative_attention.relative_attentions.0.values_mode=similarities'  \
#  'nn.module.model.relative_attention.relative_attentions.0.values_self_attention_nhead=null' \
#  'nn.module.model.relative_attention.relative_attentions.0.similarities_quantization_mode=differentiable_round'  \
#  'nn.module.model.relative_attention.relative_attentions.0.similarities_bin_size=0.1,0.2,0.3,0.5'


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


