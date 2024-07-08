import typing
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


class ListNetLoss(nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = - (target.softmax(dim=-1) * predict.log_softmax(dim=-1))
        return getattr(out, self.reduction)()


class VAEOmicsLoss(nn.Module):

    def __init__(self, loss_type: str, omics_num: int) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.k = omics_num
        self.kl_loss_weight = 0.1  # TODO: 待定

    def forward(self, x: typing.Sequence, out_x: typing.Sequence, **kwargs) -> float:
        return getattr(self, f"_forward_{self.loss_type}")(x, out_x, **kwargs)

    def _forward_generate(self, x: typing.Sequence, out_x: typing.Sequence, labels: torch.Tensor, **kwargs) -> typing.Sequence:
        # out_encoder, out_self, out_cross, out_dsc, out_cl = out_x
        out_encoder, out_self, out_cross, out_dsc = out_x
        self_loss = self._calc_self_vae_loss(x, out_self)
        cross_loss, cross_infer_dsc_loss = self._calc_cross_vae_loss(x, out_cross, out_encoder)
        cross_infer_loss = self._calc_cross_infer_loss(out_encoder)
        dsc_loss = self._calc_dsc_loss(out_dsc)
        # contrastive_loss = self._calc_contrastive_loss(out_cl, labels)
        generate_loss = (
            self_loss + 0.1 * (cross_loss + cross_infer_loss * cross_infer_loss)
            - (dsc_loss + cross_infer_dsc_loss) * 0.01  # + contrastive_loss
        )
        # return generate_loss, self_loss, cross_loss, cross_infer_loss, dsc_loss
        return generate_loss

    def _forward_dsc(self, x: typing.Sequence, out_x: typing.Sequence, **kwargs) -> float:
        out_encoder, out_cross, out_dsc = out_x
        _, cross_infer_dsc_loss = self._calc_cross_vae_loss(x, out_cross, out_encoder)
        dsc_loss = self._calc_dsc_loss(out_dsc)
        return cross_infer_dsc_loss + dsc_loss

    def _calc_self_vae_loss(self, x: typing.Sequence, out_self: typing.Sequence) -> float:
        loss = 0.
        for i, v in enumerate(out_self):
            recon_omics, mu, log_var = v
            loss += (self.kl_loss_weight * self._kl_loss(mu, log_var, 1.0) + self.reconstruction_loss(x[i], recon_omics))
        return loss

    def _calc_cross_vae_loss(self, x: typing.Sequence, out_cross: typing.Sequence, out_encoder: typing.Sequence) -> typing.Sequence:
        batch_size = x[0].size(0)
        device = x[0].device
        cross_elbo, cross_infer_loss, cross_kl_loss, cross_dsc_loss = 0, 0, 0, 0
        for i, v in enumerate(out_cross):
            _, real_mu, real_log_var = out_encoder[i][i]
            reconstruct_omic, poe_mu, poe_log_var, pred_real_modal, pred_infer_modal = v
            cross_elbo += (
                self.kl_loss_weight * self._kl_loss(poe_mu, poe_log_var, 1.0)
                + self.reconstruction_loss(x[i], reconstruct_omic)
            )
            cross_infer_loss += self.reconstruction_loss(real_mu, poe_mu)
            cross_kl_loss += self._kl_divergence(poe_mu, real_mu, poe_log_var, real_log_var)

            real_modal = torch.tensor([1 for _ in range(batch_size)]).to(device)
            infer_modal = torch.tensor([0 for _ in range(batch_size)]).to(device)
            cross_dsc_loss += torch.nn.CrossEntropyLoss()(pred_real_modal, real_modal)
            cross_dsc_loss += torch.nn.CrossEntropyLoss()(pred_infer_modal, infer_modal)

        cross_dsc_loss = cross_dsc_loss.sum(0) / (len(out_cross) * batch_size)
        return cross_elbo + cross_infer_loss + self.kl_loss_weight * cross_kl_loss, cross_dsc_loss

    def _calc_cross_infer_loss(self, out_encoder: typing.Sequence) -> float:
        infer_loss = 0
        for i in range(self.k):
            _, latent_mu, _ = out_encoder[i][i]
            for j in range(self.k):
                if i == j:
                    continue
                _, latent_mu_infer, _ = out_encoder[j][i]
                infer_loss += self.reconstruction_loss(latent_mu_infer, latent_mu)
        return infer_loss / self.k

    def _calc_dsc_loss(self, out_dsc: typing.Sequence) -> float:
        dsc_loss = 0
        batch_size = out_dsc[0].size(0)
        for i in range(self.k):
            real_modal = torch.tensor([i for _ in range(batch_size)])
            dsc_loss += torch.nn.CrossEntropyLoss()(out_dsc[i], real_modal.to(out_dsc[i].device))
        return dsc_loss.sum(0) / (self.k * batch_size)

    def _calc_contrastive_loss(self, out_cl: typing.Sequence, labels: torch.Tensor) -> float:
        margin = 1.0
        distances = torch.cdist(out_cl, out_cl)

        labels_matrix = labels.view(-1, 1) == labels.view(1, -1)

        positive_pair_distances = distances * labels_matrix.float()
        negative_pair_distances = distances * (1 - labels_matrix.float())

        positive_loss = positive_pair_distances.sum() / labels_matrix.float().sum()
        negative_loss = torch.nn.ReLU()(margin - negative_pair_distances).sum() / (1 - labels_matrix.float()).sum()

        return positive_loss + negative_loss

    def _kl_loss(self, mu, logvar, beta):
        # KL divergence loss
        kld_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return beta * kld_1

    def _kl_divergence(self, mu1, mu2, log_sigma1, log_sigma2):
        p = Normal(mu1, torch.exp(log_sigma1))
        q = Normal(mu2, torch.exp(log_sigma2))

        # 计算KL损失
        kl_loss = kl_divergence(p, q).mean()
        return kl_loss

    def reconstruction_loss(self, recon_x, x):
        # batch_size = recon_x.size(0)
        mse = nn.MSELoss()  # reduction='sum'
        recons_loss = mse(recon_x, x)  # / batch_size
        return recons_loss
