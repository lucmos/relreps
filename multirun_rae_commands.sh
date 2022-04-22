#!/bin/bash

# Variational RAE

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=True \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, anchors-class-zero]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[1, 21, 34, 37, 51, 56, 63, 68, 69, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, anchors-class-zero]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[1, 21, 34, 37, 51, 56, 63, 68, 69, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=True \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, anchors-class-zero]' \
    train.seed_index=1 \
    nn.data.anchors_idxs='[1, 21, 34, 37, 51, 56, 63, 68, 69, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, anchors-class-zero]' \
    train.seed_index=1 \
    nn.data.anchors_idxs='[1, 21, 34, 37, 51, 56, 63, 68, 69, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=True \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, anchors-class-zero]' \
    train.seed_index=2 \
    nn.data.anchors_idxs='[1, 21, 34, 37, 51, 56, 63, 68, 69, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, anchors-class-zero]' \
    train.seed_index=2 \
    nn.data.anchors_idxs='[1, 21, 34, 37, 51, 56, 63, 68, 69, 75]'
