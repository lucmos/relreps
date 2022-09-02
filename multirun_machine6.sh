#!/bin/bash

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_most_frequent,k_means_random,most_frequent,random"\
  nn.data.anchors_num=10,100,500,1000\
  train.trainer.max_epochs=10
