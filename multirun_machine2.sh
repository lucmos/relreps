#!/bin/bash

# RelResNet No Quantized
python src/rae/run.py \
    core.tags='[resnet, pretrained, relative]' \
    nn/module/model=relresnet \
    nn.module.model.finetune=False \
    nn.module.model.similarity_mode=inner \
    nn.module.model.similarities_quantization_mode=null \
    nn.module.model.similarities_bin_size=null

# RelResNet
python src/rae/run.py \
    core.tags='[resnet, pretrained, quantization, relative]' \
    nn/module/model=relresnet \
    nn.module.model.finetune=False \
    nn.module.model.similarity_mode=inner \
    nn.module.model.similarities_quantization_mode=differentiable_round \
    nn.module.model.similarities_bin_size=0.5

python src/rae/run.py \
  core.tags='[resnet, pretrained, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=False \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.25

python src/rae/run.py \
  core.tags='[resnet, pretrained, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=False \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.1

python src/rae/run.py \
  core.tags='[resnet, pretrained, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=False \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.05

python src/rae/run.py \
  core.tags='[resnet, pretrained, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=False \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.005

# ResNet
python src/rae/run.py \
  core.tags='[resnet, pretrained, absolute]' \
  nn/module/model=resnet \
  nn.module.model.finetune=False \
