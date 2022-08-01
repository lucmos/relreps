#!/bin/bash

python src/rae/run.py -m \
  core.tags='[classification, relative, subspaces, repr_pooling]' \
  nn.data.anchors_num=500 \
  nn/data/datasets=vision/cifar100 \
  nn/module=classifier \
  nn/module/model=rcnn \
  nn.module.model.hidden_features=512 \
  nn.module.model.values_mode=similarities \
  nn.module.model.num_subspaces='1,8' \
  nn.module.model.repr_pooling='max,sum,linear,mean,none' \
  nn.module.model.values_self_attention_nhead=null \
  nn.module.model.transform_elements=null \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.similarities_quantization_mode=null \
  nn.module.model.similarities_bin_size=null \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=null \
  nn.module.model.n_anchors_sampling_per_class=null \
  nn.module.model.dropout_p=0.1
