import torch
import torch.nn.functional as F


class VaeLoss:
    def __init__(self, variational_beta: float):
        self.variational_beta = variational_beta

    def __call__(self, image_pred: torch.Tensor, image_batch: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        # recon_loss = F.binary_cross_entropy(image_pred.view(-1, 784), image_batch.view(-1, 784), reduction="sum")
        #
        recon_loss = F.mse_loss(
            image_pred,
            image_batch,
            reduction="sum",
        )

        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + self.variational_beta * kldivergence
