#!/bin/bash

# Variational RAE

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=True \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, same-class-anchors]' \
    train.seed_index=0 \
    nn.data.anchors_idx='[0, 17, 26, 34, 36, 41, 60, 64, 70, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, same-class-anchors]' \
    train.seed_index=0 \
    nn.data.anchors_idx='[0, 17, 26, 34, 36, 41, 60, 64, 70, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=True \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, same-class-anchors]' \
    train.seed_index=1 \
    nn.data.anchors_idx='[0, 17, 26, 34, 36, 41, 60, 64, 70, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, same-class-anchors]' \
    train.seed_index=1 \
    nn.data.anchors_idx='[0, 17, 26, 34, 36, 41, 60, 64, 70, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=True \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, same-class-anchors]' \
    train.seed_index=2 \
    nn.data.anchors_idx='[0, 17, 26, 34, 36, 41, 60, 64, 70, 75]'

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, same-class-anchors]' \
    train.seed_index=2 \
    nn.data.anchors_idx='[0, 17, 26, 34, 36, 41, 60, 64, 70, 75]'
