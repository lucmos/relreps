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
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20 \

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=0 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20



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
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=1 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=1 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20


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
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=2 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=2 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20


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
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=3 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=3 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20


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
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=4 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=4 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20


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
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=5 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=5 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20

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
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=6 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20

python src/rae/run.py \
    nn.module._target_=rae.pl_modules.pl_variational.LightningVariational \
    nn.module.autoencoder._target_=rae.modules.rae.RAE \
    nn.module.autoencoder.reparametrize_anchors=False \
    nn.module.autoencoder.normalize_latents=off \
    nn.module.autoencoder.normalize_only_anchors_latents=True \
    nn.module.autoencoder.normalize_means=l2 \
    nn.module.autoencoder.normalize_only_anchors_means=True \
    nn.module.autoencoder.normalize_relative_embedding='off' \
    nn.module.autoencoder.relative_embedding_method=basis_change \
    core.tags='[rae, normalize-only-anchors-means, no-lr-scheduler]' \
    train.seed_index=6 \
    nn.data.anchors_mode=fixed \
    nn.data.anchors_idxs='[0, 1, 2, 3, 4, 5, 7, 13, 15, 17]' \
    nn.module.autoencoder.latent_dim=20
