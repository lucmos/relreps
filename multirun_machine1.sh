#!/bin/bash

# Variational RAE

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=2 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=3 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=4 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=5 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=6 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=7 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=8 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=9 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=10 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=15 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=30 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=35 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=40 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=50 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=60 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=75 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=90 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=110 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=130 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=150 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=200 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=250 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=300 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=350 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=400 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=450 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=500 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=750 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=fixed \
    core.tags='[rae, latent-dim-sweep, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=1000 \
