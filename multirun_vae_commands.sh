#!/bin/bash

# Variational vae
python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.vae.VAE \
    nn.data.anchors_mode=random_latents \
    core.tags='[vae]' \
    train.seed_index=0

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.vae.VAE \
    nn.data.anchors_mode=random_latents \
    core.tags='[vae]' \
    train.seed_index=1

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.vae.VAE \
    nn.data.anchors_mode=random_latents \
    core.tags='[vae]' \
    train.seed_index=2
