#!/bin/bash

# Variational RAE
python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.normalize_latents=layernorm \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=2

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.normalize_latents=layernorm \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=10

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.normalize_latents=layernorm \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20


# Variational RAE
python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.normalize_latents=layernorm \
    nn.module.autoencoder.normalize_only_anchors_latents=False \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=2

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.normalize_latents=layernorm \
    nn.module.autoencoder.normalize_only_anchors_latents=False \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=10

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.normalize_latents=layernorm \
    nn.module.autoencoder.normalize_only_anchors_latents=False \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20
