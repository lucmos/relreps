#!/bin/bash

# Variational RAE
python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_gclassifier.LightningClassifier \
    nn.module.model._target_=rae.modules.relative_classifier.RCNN \
    nn.module.model.hidden_channels=64 \
    nn.module.model.hidden_features=16 \
    nn.module.model.normalization_mode=l2 \
    nn.module.model.similarity_mode=basis_change \
    nn.module.model.values_mode=similarities \
    core.tags='[classifier, rcnn]' \
    nn.data.anchors_num=500 \
    nn.data.anchors_mode=stratified_subset
