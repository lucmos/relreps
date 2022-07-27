#!/bin/bash

python src/rae/run.py \
  core.tags='[complete-bootstrap,  memory, continual, absolute]' \
  nn.data.anchors_num=500 \
  nn/data/datasets=continual/cifar10 \
  nn/module=continual_classifier \
  nn/module/model=cnn \
  nn.data.datasets.tasks_epochs="[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]" \
  nn.data.datasets.tasks_progression="[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  [0, 1],  [1, 2],  [2, 3],  [3, 4],  [4, 5],  [5, 6],  [6, 7],  [7, 8],  [8, 9]]" \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.replay.max_size=0 \
  nn.module.memory.limit_target_representation=False \
  nn.module.memory.start_epoch=9 \
  nn.module.memory.loss_weight=1e6
