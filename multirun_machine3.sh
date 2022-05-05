#!/bin/bash

# Variational RAE

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=2 \
    nn.module.autoencoder.latent_dim=2 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=3 \
    nn.module.autoencoder.latent_dim=3 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=4 \
    nn.module.autoencoder.latent_dim=4 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=5 \
    nn.module.autoencoder.latent_dim=5 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=6 \
    nn.module.autoencoder.latent_dim=6 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=7 \
    nn.module.autoencoder.latent_dim=7 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=8 \
    nn.module.autoencoder.latent_dim=8 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=9 \
    nn.module.autoencoder.latent_dim=9 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=10 \
    nn.module.autoencoder.latent_dim=10 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=15 \
    nn.module.autoencoder.latent_dim=15 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=20 \
    nn.module.autoencoder.latent_dim=20 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=30 \
    nn.module.autoencoder.latent_dim=30 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=35 \
    nn.module.autoencoder.latent_dim=35 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=40 \
    nn.module.autoencoder.latent_dim=40 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=50 \
    nn.module.autoencoder.latent_dim=50 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=60 \
    nn.module.autoencoder.latent_dim=60 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=75 \
    nn.module.autoencoder.latent_dim=75 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=90 \
    nn.module.autoencoder.latent_dim=90 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=110 \
    nn.module.autoencoder.latent_dim=110 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=130 \
    nn.module.autoencoder.latent_dim=130 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=150 \
    nn.module.autoencoder.latent_dim=150 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=200 \
    nn.module.autoencoder.latent_dim=200 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=250 \
    nn.module.autoencoder.latent_dim=250 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=300 \
    nn.module.autoencoder.latent_dim=300 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=350 \
    nn.module.autoencoder.latent_dim=350 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=400 \
    nn.module.autoencoder.latent_dim=400 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=450 \
    nn.module.autoencoder.latent_dim=450 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=500 \
    nn.module.autoencoder.latent_dim=500 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=750 \
    nn.module.autoencoder.latent_dim=750 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    +nn.module.autoencoder.normalize_latents=False \
    nn.data.anchors_mode=stratified \
    core.tags='[rae, latent-as-anchors, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_idxs=null \
    nn.data.anchors_num=1000 \
    nn.module.autoencoder.latent_dim=1000 \
