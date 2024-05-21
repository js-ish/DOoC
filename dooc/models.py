import torch
from torch import nn
from dooc import nets
from moltx.models import AdaMR
from moltx.nets import AbsPosEncoderDecoderConfig


class MutSmi(nn.Module):
    """Base"""

    def __init__(
        self,
        gene_conf: nets.GeneGNNConfig = nets.GeneGNN.DEFAULT_CONFIG,
        smiles_conf: AbsPosEncoderDecoderConfig = AdaMR.CONFIG_BASE,
    ) -> None:
        super().__init__()
        self.gene_conf = gene_conf
        self.smiles_conf = smiles_conf
        self.smiles_encoder = AdaMR(smiles_conf)

        self.gene_encoder = nets.GeneGNN(gene_conf)

    def load_ckpt(self, *ckpt_files: str) -> None:
        """load check point model.

        Args:
            ckpt_files (str): check point file paths.
        """
        self.load_state_dict(
            torch.load(ckpt_files[0], map_location=torch.device("cpu")), strict=False
        )

    def load_pretrained_ckpt(self, drugcell_ckpt: str, moltx_ckpt: str, freeze_drugcell: bool = False, freeze_moltx: bool = False) -> None:
        self.gene_encoder.load_ckpt(drugcell_ckpt)
        self.smiles_encoder.load_ckpt(moltx_ckpt)
        if freeze_moltx:
            self.smiles_encoder.requires_grad_(False)
        if freeze_drugcell:
            self.gene_encoder.requires_grad_(False)

    def forward(
        self, smiles_src: torch.Tensor, smiles_tgt: torch.Tensor, gene_src: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()


class MutSmiXAttention(MutSmi):
    """Regression model using transformer cross attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 2,
        num_layers: int = 2,
        gene_conf: nets.GeneGNNConfig = nets.GeneGNN.DEFAULT_CONFIG,
        smiles_conf: AbsPosEncoderDecoderConfig = AdaMR.CONFIG_BASE,
    ) -> None:
        super().__init__(gene_conf, smiles_conf)
        self.d_model = d_model
        d_hidden = d_model // 2
        self.nhead = nhead
        self.num_layers = num_layers
        self.gene_fc = nn.Linear(self.gene_conf.num_hiddens_genotype, self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(self.d_model,
                                                   self.nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=self.num_layers)
        self.reg = nn.Sequential(
            nn.Linear(self.d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self, smiles_src: torch.Tensor, smiles_tgt: torch.Tensor, gene_src: torch.Tensor
    ) -> torch.Tensor:
        is_batched = smiles_src.dim() == 2

        smiles_out = self.smiles_encoder.forward_feature(smiles_src, smiles_tgt)
        gene_out = self.gene_fc(self.gene_encoder(gene_src))
        feat = None
        if is_batched:
            feat = self.decoder(smiles_out, gene_out)
        else:
            feat = self.decoder(smiles_out.unsqueeze(0), gene_out)

        return self.reg(feat)


class MutSmiFullConnection(MutSmi):
    """Regression model using fully connection."""

    def __init__(
        self,
        d_model: int,
        gene_conf: nets.GeneGNNConfig = nets.GeneGNN.DEFAULT_CONFIG,
        smiles_conf: AbsPosEncoderDecoderConfig = AdaMR.CONFIG_BASE,
    ) -> None:
        super().__init__(gene_conf, smiles_conf)
        self.d_model = d_model

        d_hidden = d_model // 2
        self.gene_fc = nn.Linear(self.gene_conf.num_hiddens_genotype, self.d_model)
        self.reg = nn.Sequential(
            nn.Linear(self.d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self, smiles_src: torch.Tensor, smiles_tgt: torch.Tensor, gene_src: torch.Tensor
    ) -> torch.Tensor:
        is_batched = smiles_src.dim() == 2

        smiles_out = self.smiles_encoder.forward_feature(smiles_src, smiles_tgt)
        gene_out = self.gene_encoder(gene_src).unsqueeze(1)

        feat = None
        if is_batched:
            feat = smiles_out + self.gene_fc(gene_out)[:, 0]
        else:
            feat = smiles_out[0] + self.gene_fc(gene_out)[0]

        return self.reg(feat)
