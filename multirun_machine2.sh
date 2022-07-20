#!/bin/bash

python src/rae/run.py \
  core.tags='[continual, absolute]' \
  nn.data.anchors_num=500 \
  nn/data/datasets=continual/cifar10 \
  nn/module=continual_classifier \
  nn/module/model=cnn \
  nn.module.model.hidden_channels=512 \
  nn.module.model.dropout_p=0.5
