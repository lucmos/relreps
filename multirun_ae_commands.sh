#!/bin/bash

# Variational ae

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=fixed \
    core.tags='[ae]' \
    train.seed_index=0

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=random_images \
    core.tags='[ae]' \
    train.seed_index=0

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=random_latents \
    core.tags='[ae]' \
    train.seed_index=0

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=fixed \
    core.tags='[ae]' \
    train.seed_index=1

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=random_images \
    core.tags='[ae]' \
    train.seed_index=1

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=random_latents \
    core.tags='[ae]' \
    train.seed_index=1

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=fixed \
    core.tags='[ae]' \
    train.seed_index=2

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=random_images \
    core.tags='[ae]' \
    train.seed_index=2

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_deterministic.LightningDeterministic \
    nn.module.autoencoder._target_=rae.modules.ae.AE \
    nn.data.anchors_mode=random_latents \
    core.tags='[ae]' \
    train.seed_index=2
