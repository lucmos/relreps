#!/bin/bash

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, keys-queries-transform]' \
  nn.data.anchors_num=500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.transform_elements='[attention_keys, attention_queries]' \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.5 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=null \
  nn.module.model.n_anchors_sampling_per_class=null

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, keys-queries-transform]' \
  nn.data.anchors_num=500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.transform_elements='[attention_keys, attention_queries]' \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.1 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=null \
  nn.module.model.n_anchors_sampling_per_class=null

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, relative, keys-queries-transform]' \
  nn.data.anchors_num=500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.transform_elements='[attention_keys, attention_queries]' \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=null \
  nn.module.model.similarities_bin_size=null \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=null \
  nn.module.model.n_anchors_sampling_per_class=null

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling, keys-queries-transform]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.transform_elements='[attention_keys, attention_queries]' \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.5 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling, keys-queries-transform]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.transform_elements='[attention_keys, attention_queries]' \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.1 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, relative, anchors-sampling, keys-queries-transform]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.transform_elements='[attention_keys, attention_queries]' \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=null \
  nn.module.model.similarities_bin_size=null \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3
