#!/bin/bash


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=True \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.1 \
  nn.module.model.transform_resnet_features=True

# ResNet
python src/rae/run.py \
  core.tags='[resnet, pretrained, absolute]' \
  nn/module/model=resnet \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=True \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.5 \
  nn.module.model.transform_resnet_features=True

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=True \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.25 \
  nn.module.model.transform_resnet_features=True

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=True \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.05 \
  nn.module.model.transform_resnet_features=True

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=True \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.005 \
  nn.module.model.transform_resnet_features=True
