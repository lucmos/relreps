#!/bin/bash

# Variational RAE
python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_gclassifier.LightningClassifier \
    nn.module.model._target_=rae.modules.classifier.CNN \
    nn.module.model.hidden_channels=64 \
    core.tags='[classifier, cnn]' \
