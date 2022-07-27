#!/bin/bash



# Relative

python src/rae/run.py \
  core.tags='[complete-bootstrap, memory, continual, relative]' \
  nn.data.anchors_num=500 \
  nn/data/datasets=continual/cifar10 \
  nn/module=continual_classifier \
  nn/module/model=rcnn \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.replay.max_size=0 \
  nn.module.memory.limit_target_representation=True \
  nn.module.memory.start_epoch=9 \
  nn.module.memory.loss_weight=1e6

python src/rae/run.py \
  core.tags='[complete-bootstrap, memory, continual, relative]' \
  nn.data.anchors_num=500 \
  nn/data/datasets=continual/cifar10 \
  nn/module=continual_classifier \
  nn/module/model=rcnn \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.replay.max_size=0 \
  nn.module.memory.limit_target_representation=False \
  nn.module.memory.start_epoch=9 \
  nn.module.memory.loss_weight=1e6

#python src/rae/run.py \
#  core.tags='[complete-bootstrap, continual, relative]' \
#  nn.data.anchors_num=500 \
#  nn/data/datasets=continual/cifar10 \
#  nn/module=continual_classifier \
#  nn/module/model=rcnn \
#  nn.module.model.hidden_features=512 \
#  nn.module.model.dropout_p=0.5 \
#  nn.module.replay.max_size=100
#
#python src/rae/run.py \
#  core.tags='[complete-bootstrap, continual, relative]' \
#  nn.data.anchors_num=500 \
#  nn/data/datasets=continual/cifar10 \
#  nn/module=continual_classifier \
#  nn/module/model=rcnn \
#  nn.module.model.similarities_quantization_mode=differentiable_round \
#  nn.module.model.similarities_bin_size=0.5 \
#  nn.module.model.hidden_features=512 \
#  nn.module.model.dropout_p=0.5 \
#  nn.module.replay.max_size=0
#
#python src/rae/run.py \
#  core.tags='[complete-bootstrap, continual, relative]' \
#  nn.data.anchors_num=500 \
#  nn/data/datasets=continual/cifar10 \
#  nn/module=continual_classifier \
#  nn/module/model=rcnn \
#  nn.module.model.similarities_quantization_mode=differentiable_round \
#  nn.module.model.similarities_bin_size=0.5 \
#  nn.module.model.hidden_features=512 \
#  nn.module.model.dropout_p=0.5 \
#  nn.module.replay.max_size=100

# Absolute

python src/rae/run.py \
  core.tags='[complete-bootstrap,  memory, continual, absolute]' \
  nn.data.anchors_num=500 \
  nn/data/datasets=continual/cifar10 \
  nn/module=continual_classifier \
  nn/module/model=cnn \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.replay.max_size=0 \
  nn.module.memory.limit_target_representation=False \
  nn.module.memory.start_epoch=9 \
  nn.module.memory.loss_weight=1e6

#python src/rae/run.py \
#  core.tags='[complete-bootstrap, continual, absolute]' \
#  nn.data.anchors_num=500 \
#  nn/data/datasets=continual/cifar10 \
#  nn/module=continual_classifier \
#  nn/module/model=cnn \
#  nn.module.model.hidden_features=512 \
#  nn.module.model.dropout_p=0.5 \
#  nn.module.replay.max_size=100
