import torch
import typing
import torch.nn as nn
from dataclasses import dataclass


def product_of_experts(mu_set, log_var_set):
    tmp = 0
    for i in range(len(mu_set)):
        tmp += torch.div(1, torch.exp(log_var_set[i]))

    poe_var = torch.div(1., tmp)
    poe_log_var = torch.log(poe_var)

    tmp = 0.
    for i in range(len(mu_set)):
        tmp += torch.div(1., torch.exp(log_var_set[i])) * mu_set[i]
    poe_mu = poe_var * tmp
    return poe_mu, poe_log_var


def reparameterize(mean, logvar):
    std = torch.exp(logvar / 2)
    epsilon = torch.randn_like(std)
    return epsilon * std + mean


class LinearLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2, batchnorm: bool = False, activation=None) -> None:
        super(LinearLayer, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batchnorm = nn.BatchNorm1d(output_dim) if batchnorm else None

        self.activation = None
        if activation is not None:
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class VAEEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: typing.Sequence) -> None:
        super(VAEEncoder, self).__init__()
        self.encoders = nn.ModuleList([
            LinearLayer(
                input_dim, hidden_dims[0],
                batchnorm=True, activation='relu'
            )
        ])
        for i in range(len(hidden_dims) - 1):
            self.encoders.append(
                LinearLayer(
                    hidden_dims[i], hidden_dims[i + 1],
                    batchnorm=True, activation='relu'
                )
            )

        self.mu_predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim), nn.ReLU()
        )
        self.log_var_predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim), nn.ReLU()
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def forward(self, x: torch.Tensor) -> typing.Sequence:
        for layer in self.encoders:
            x = layer(x)
        mu = self.mu_predictor(x)
        log_var = self.log_var_predictor(x)
        latent_z = self.reparameterize(mu, log_var)
        return latent_z, mu, log_var


class VAEDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: typing.Sequence) -> None:
        super(VAEDecoder, self).__init__()

        self.decoders = nn.ModuleList([LinearLayer(
            input_dim, hidden_dims[0],
            dropout=0.1, batchnorm=True,
            activation='relu'
        )])
        for i in range(len(hidden_dims) - 1):
            self.decoders.append(LinearLayer(
                hidden_dims[i], hidden_dims[i + 1],
                dropout=0.1, batchnorm=True,
                activation='relu'
            ))

        self.recons_predictor = LinearLayer(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoders:
            x = layer(x)
        data_recons = self.recons_predictor(x)
        return data_recons


@dataclass
class VAEOmicsConfig:
    modal_num: int
    modal_dim: list[int]
    latent_dim: int
    encoder_hidden_dims: list[int]
    decoder_hidden_dims: list[int]
    complete_omics: dict[str]


class VAEOmics(nn.Module):
    DEFAULT_CONFIG = VAEOmicsConfig(
        modal_num=3,
        modal_dim=[2827, 2739, 8802],
        latent_dim=256,
        encoder_hidden_dims = [2048, 512],
        decoder_hidden_dims = [512, 2048],
        complete_omics={'methylation': 0, 'mutation': 1, 'rna': 2},
    )

    def __init__(self, config: VAEOmicsConfig) -> None:
        super(VAEOmics, self).__init__()

        self.config = config
        self.device = torch.device('cpu')

        self.k = config.modal_num
        self.encoders = nn.ModuleList(
            nn.ModuleList([
                VAEEncoder(
                    self.config.modal_dim[i], self.config.latent_dim, self.config.encoder_hidden_dims
                ) for j in range(self.k)
            ]) for i in range(self.k)
        )

        self.self_decoders = nn.ModuleList([
            VAEDecoder(
                self.config.latent_dim, self.config.modal_dim[i], self.config.decoder_hidden_dims
            ) for i in range(self.k)
        ])

        self.cross_decoders = nn.ModuleList([
            VAEDecoder(
                self.config.latent_dim, self.config.modal_dim[i], self.config.decoder_hidden_dims
            ) for i in range(self.k)
        ])

        self.share_encoder = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.latent_dim),
            nn.BatchNorm1d(self.config.latent_dim),
            nn.ReLU()
        )

        self.discriminator = nn.Sequential(
            nn.Linear(self.config.latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, self.k)
        )

        self.infer_discriminator = nn.ModuleList(nn.Sequential(
            nn.Linear(self.config.latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 2)
        ) for i in range(self.k))

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(
            torch.load(ckpt_files[0], map_location=torch.device("cpu")), strict=False
        )

    def freeze_switch(self, layer_name: str, freeze: bool = False) -> None:
        layer = getattr(self, layer_name)

        def switch(model, freeze: bool = False) -> None:
            for _, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = not freeze
                switch(child)

        switch(layer, freeze)

    def forward_generate(self, *x: torch.Tensor) -> typing.Sequence:
        '''
        x: list of omics tensor
        '''
        self.device = x[0].device
        output = [[0 for _ in range(self.k)] for _ in range(self.k)]
        for (idx, i) in enumerate(range(self.k)):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](x[idx])

        se = [output[i][i] for i in range(self.k)]
        out_self = self._forward_self_vae(se)
        out_cross = self._forward_cross_vae(output)
        out_disc = self._forward_discriminator(output)
        # out_cl = self._forward_contrastive_learning(*x)
        return output, out_self, out_cross, out_disc  # , out_cl

    def forward_dsc(self, *x: torch.Tensor) -> typing.Sequence:
        self.device = x[0].device
        output = [[0 for _ in range(self.k)] for _ in range(self.k)]
        for (idx, i) in enumerate(range(self.k)):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](x[idx])

        out_cross = self._forward_cross_vae(output)
        out_disc = self._forward_discriminator(output)
        return output, out_cross, out_disc

    # can input incomplete omics
    def forward_encoder(self, omics: dict, *x: torch.Tensor) -> torch.Tensor:
        self.device = x[0].device
        values = list(omics.values())
        output = [[0 for _ in range(self.k)] for _ in range(self.k)]

        for (item, i) in enumerate(values):
            for j in range(self.k):
                output[i][j] = self.encoders[i][j](x[item])

        embedding_tensor = []
        for i in range(self.k):
            mu_set = []
            log_var_set = []
            for j in range(self.k):
                if i == j or j not in values:
                    continue
                _, mu, log_var = output[j][i]
                mu_set.append(mu)
                log_var_set.append(log_var)
            poe_mu, _ = product_of_experts(mu_set, log_var_set)
            if i in values:
                _, omic_mu, _ = output[i][i]
                joint_mu = (omic_mu + poe_mu) / 2
            else:
                joint_mu = poe_mu
            embedding_tensor.append(joint_mu.to(self.device))
        embedding_tensor = torch.cat(embedding_tensor, dim=1)
        return embedding_tensor

    def _forward_self_vae(self, se: typing.Sequence) -> typing.Sequence:
        output = []
        for i in range(self.k):
            latent_z, mu, log_var = se[i]
            recon_omics = self.self_decoders[i](latent_z)
            output.append((recon_omics, mu, log_var))
        return output

    def _forward_cross_vae(self, e: typing.Sequence) -> typing.Sequence:
        output = []
        for i in range(self.k):
            _, real_mu, _ = e[i][i]
            mus = []
            log_vars = []
            for j in range(self.k):
                _, mu, log_var = e[j][i]
                mus.append(mu)
                log_vars.append(log_var)
            poe_mu, poe_log_var = product_of_experts(mus, log_vars)
            poe_mu = poe_mu.to(self.device)
            poe_log_var = poe_log_var.to(self.device)
            poe_latent_z = reparameterize(poe_mu, poe_log_var).to(self.device)
            reconstruct_omic = self.self_decoders[i](poe_latent_z)

            pred_real_modal = self.infer_discriminator[i](real_mu)
            pred_infer_modal = self.infer_discriminator[i](poe_mu)

            output.append((reconstruct_omic, poe_mu, poe_log_var, pred_real_modal, pred_infer_modal))
        return output

    def _forward_discriminator(self, e: typing.Sequence) -> typing.Sequence:
        output = []
        for i in range(self.k):
            _, mu, _ = e[i][i]
            pred_modal = self.discriminator(mu)
            output.append(pred_modal)
        return output

    def _forward_contrastive_learning(self, *x: torch.Tensor) -> typing.Sequence:
        return self.forward_encoder(self.config.complete_omics, *x)
