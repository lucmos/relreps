#!/bin/bash

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_most_frequent"\
  nn.data.anchors_num=10 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_random"\
  nn.data.anchors_num=10 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="most_frequent"\
  nn.data.anchors_num=10 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="random"\
  nn.data.anchors_num=10 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_most_frequent"\
  nn.data.anchors_num=100 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_random"\
  nn.data.anchors_num=100 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="most_frequent"\
  nn.data.anchors_num=100 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="random"\
  nn.data.anchors_num=100 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_most_frequent"\
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_random"\
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="most_frequent"\
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="random"\
  nn.data.anchors_num=500 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_most_frequent"\
  nn.data.anchors_num=1000 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="k_means_random"\
  nn.data.anchors_num=1000 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="most_frequent"\
  nn.data.anchors_num=1000 \
  train.trainer.max_epochs=10

python src/rae/run.py -m \
  core.tags='[semantic_horizon, develop, word_anchors, anchors_num, anchors_method]' \
  'train.seed_index=range(0, 4)' \
  nn.data.datasets.anchors.method="random"\
  nn.data.anchors_num=1000 \
  train.trainer.max_epochs=10
