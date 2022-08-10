#!/bin/bash
# Relative
python src/rae/run.py -m \
  core.tags='[classification, relative]' \
  nn/data/datasets=vision/cifar10 \
  nn/module=classifier \
  nn/module/model=vision/relresnet \
  nn.module.model.resnet_size=18 \
  nn.module.model.input_size=224 \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=False \
  nn.module.model.hidden_features=512 \
  'nn.module.model.relative_attention.output_normalization_mode=none,l2,batchnorm,layernorm,instancenorm' \
  train.trainer.max_epochs=10

# Absolute
#python src/rae/run.py -m \
#  core.tags='[classification, absolute]' \
#  nn/data/datasets=vision/cifar10 \
#  nn/module=classifier \
#  nn/module/model=vision/resnet \
#  nn.module.model.resnet_size=18 \
#  nn.module.model.input_size=224 \
#  nn.module.model.use_pretrained=True \
#  nn.module.model.finetune=False \
#  nn.module.model.hidden_features=512


#python src/rae/run.py -m \
#  core.tags='[classification, absolute]' \
#  nn.data.anchors_num=500 \
#  nn/data/datasets=vision/cifar10 \
#  nn/module=classifier \
#  nn/module/model=vision/seresnet

#  nn.module.model.hidden_features=512 \
#  nn.module.model.values_mode=similarities \
#  nn.module.model.num_subspaces='1,8' \
#  nn.module.model.repr_pooling='max,sum,linear,mean,none' \
#  nn.module.model.values_self_attention_nhead=null \
#  nn.module.model.transform_elements=null \
#  nn.module.model.normalization_mode=l2 \
#  nn.module.model.similarity_mode=inner \
#  nn.module.model.similarities_quantization_mode=null \
#  nn.module.model.similarities_bin_size=null \
#  nn.module.model.similarities_aggregation_mode=null \
#  nn.module.model.similarities_aggregation_n_groups=null \
#  nn.module.model.anchors_sampling_mode=null \
#  nn.module.model.n_anchors_sampling_per_class=null \
#  nn.module.model.dropout_p=0.1
