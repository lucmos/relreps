import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, variational_beta):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction="sum")

    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence
