#!/bin/bash


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative]' \
  nn/module/model=relresnet \
  nn.module.model.finetune=True \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.1
  nn.module.model.transform_resnet_features=True
