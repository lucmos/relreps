#!/bin/bash

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, relative, anchors-sampling]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=null \
  nn.module.model.similarities_bin_size=null \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, relative, anchors-sampling]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=null \
  nn.module.model.similarities_bin_size=null \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, relative, anchors-sampling]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=null \
  nn.module.model.similarities_bin_size=null \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1

python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, relative, anchors-sampling]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=null \
  nn.module.model.similarities_bin_size=null \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1


# Quantized

# 0.005
python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.005 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.005 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.005 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.005 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1

# 0.1
python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
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
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.1 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.1 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.1 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1

# 0.25
python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.25 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.25 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.25 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.25 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1

# 0.5
python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
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
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.5 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=3


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=similarities \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.5 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1


python src/rae/run.py \
  core.tags='[resnet, pretrained, finetuning, quantization, relative, anchors-sampling ]' \
  nn.data.anchors_num=1500 \
  nn/module/model=relresnet \
  nn.module.model.use_pretrained=True \
  nn.module.model.finetune=True \
  nn.module.model.transform_resnet_features=True \
  nn.module.model.hidden_features=512 \
  nn.module.model.dropout_p=0.5 \
  nn.module.model.normalization_mode=l2 \
  nn.module.model.similarity_mode=inner \
  nn.module.model.values_mode=trainable \
  nn.module.model.similarities_quantization_mode=differentiable_round \
  nn.module.model.similarities_bin_size=0.5 \
  nn.module.model.similarities_aggregation_mode=null \
  nn.module.model.similarities_aggregation_n_groups=null \
  nn.module.model.anchors_sampling_mode=stratified \
  nn.module.model.n_anchors_sampling_per_class=1
