#!/bin/bash

# Variational RAE
python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=False \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae]' \
    train.seed_index=0 \
    nn.data.anchors_mode=stratified_subset \
    nn.data.anchors_num=50 \
    nn.data.anchors_idxs=null \
    nn.module.autoencoder.latent_dim=20

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=False \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=False \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae]' \
    train.seed_index=1 \
    nn.data.anchors_mode=stratified_subset \
    nn.data.anchors_num=50 \
    nn.data.anchors_idxs=null \
    nn.module.autoencoder.latent_dim=20
