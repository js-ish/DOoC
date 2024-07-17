import torch
import typing
from torch import nn
from moltx.models import AdaMR, AdaMR2
from dooc.nets.drugcell import Drugcell
from dooc.nets.vaeomic import VAEOmics


"""
Mutations(Individual Sample) and Smiles Interaction

{MutationEnc}{SmileEnc}MutSmi{Add/Xattn}: 1 mut with 1 smi
{MutationEnc}{SmileEnc}MutSmis{Add/Xattn}: 1 mut with n smi
{MutationEnc}{SmileEnc}MutsSmi{Add/Xattn}: n mut with 1 smi
"""


class _DrugcellAdamr(nn.Module):

    def __init__(self, mut_conf, smi_conf) -> None:
        super().__init__()
        self.mut_conf = mut_conf
        self.smi_conf = smi_conf

        self.mut_encoder = Drugcell(mut_conf)
        self.smi_encoder = AdaMR(smi_conf)

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(
            torch.load(ckpt_files[0], map_location=torch.device("cpu"))
        )

    def load_pretrained_ckpt(self, mut_ckpt: str, smi_ckpt: str, freeze_mut: bool = False, freeze_smi: bool = False) -> None:
        self.mut_encoder.load_ckpt(mut_ckpt)
        self.smi_encoder.load_ckpt(smi_ckpt)
        if freeze_smi:
            self.smi_encoder.requires_grad_(False)
        if freeze_mut:
            self.mut_encoder.requires_grad_(False)


class DrugcellAdamrMutSmiAdd(_DrugcellAdamr):

    def forward(
            self, mut_x: torch.Tensor, smi_src: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        """
        mut_x: [b, mut_seqlen]
        smi_src, smi_tgt: [b, smi_seqlen]
        """
        mut_out = self.mut_encoder(mut_x)
        smi_out = self.smi_encoder.forward_feature(smi_src, smi_tgt)
        return mut_out + smi_out  # [b, dmodel]


class DrugcellAdamrMutSmiXattn(_DrugcellAdamr):

    def __init__(self, mut_conf, smi_conf, nhead: int = 2, num_layers: int = 2) -> None:
        super().__init__(mut_conf, smi_conf)
        d_model = self.smi_conf.d_model
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(layer, num_layers)

    def forward(
            self, mut_x: torch.Tensor, smi_src: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        """
        mut_x: [b, mut_seqlen]
        smi_src, smi_tgt: [b, smi_seqlen]
        """
        mut_out = self.mut_encoder(mut_x).unsqueeze(-2)  # [b, 1, dmodel]
        smi_out = self.smi_encoder.forward_feature(smi_src, smi_tgt).unsqueeze(-2)  # [b, 1, dmodel]
        return self.cross_attn(smi_out, mut_out).squeeze(-2)  # [b, dmodel]


class DrugcellAdamrMutSmisAdd(_DrugcellAdamr):

    def _forward_mut(self, mut_x: torch.Tensor) -> torch.Tensor:
        """
        mut_x: [b, mut_seqlen]
        out: [b, 1, dmodel]
        """
        return self.mut_encoder(mut_x).unsqueeze(-2)

    def _forward_smi(self, smi_src: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        """
        smi_src: [b, n, smi_seqlen]
        smi_tgt: [b, n, smi_seqlen]
        out: [b, n, dmodel]
        """
        batched = smi_src.dim() == 3
        if batched:
            n = smi_src.shape[1]
            smi_src = smi_src.reshape(-1, smi_src.shape[-1])
            smi_tgt = smi_tgt.reshape(-1, smi_tgt.shape[-1])
            out = self.smi_encoder.forward_feature(smi_src, smi_tgt)
            return out.reshape(-1, n, out.shape[-1])
        return self.smi_encoder.forward_feature(smi_src, smi_tgt)

    def forward(
            self, mut_x: torch.Tensor, smi_src: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        mut_out = self._forward_mut(mut_x)
        smi_out = self._forward_smi(smi_src, smi_tgt)
        return smi_out + mut_out  # [b, n, dmodel]


class DrugcellAdamrMutSmisXattn(DrugcellAdamrMutSmisAdd):

    def __init__(self, mut_conf, smi_conf, nhead: int = 2, num_layers: int = 2) -> None:
        super().__init__(mut_conf, smi_conf)
        d_model = smi_conf.d_model
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(layer, num_layers)

    def forward(
            self, mut_x: torch.Tensor, smi_src: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        mut_out = self._forward_mut(mut_x)
        smi_out = self._forward_smi(smi_src, smi_tgt)
        return self.cross_attn(smi_out, mut_out)  # [b, n, dmodel]


class _DrugcellAdamr2(nn.Module):

    def __init__(self, mut_conf, smi_conf) -> None:
        super().__init__()
        self.mut_conf = mut_conf
        self.smi_conf = smi_conf

        self.mut_encoder = Drugcell(mut_conf)
        self.smi_encoder = AdaMR2(smi_conf)

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(
            torch.load(ckpt_files[0], map_location=torch.device("cpu"))
        )

    def load_pretrained_ckpt(self, mut_ckpt: str, smi_ckpt: str, freeze_mut: bool = False, freeze_smi: bool = False) -> None:
        self.mut_encoder.load_ckpt(mut_ckpt)
        self.smi_encoder.load_ckpt(smi_ckpt)
        if freeze_smi:
            self.smi_encoder.requires_grad_(False)
        if freeze_mut:
            self.mut_encoder.requires_grad_(False)


class DrugcellAdamr2MutSmiAdd(_DrugcellAdamr2):
    def forward(
            self, mut_x: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        """
        mut_x: [b, mut_seqlen]
        smi_tgt: [b, smi_seqlen]
        """
        mut_out = self.mut_encoder(mut_x)
        smi_out = self.smi_encoder.forward_feature(smi_tgt)
        return mut_out + smi_out  # [b, dmodel]


class DrugcellAdamr2MutSmiXattn(_DrugcellAdamr2):
    def __init__(self, mut_conf, smi_conf, nhead: int = 2, num_layers: int = 2) -> None:
        super().__init__(mut_conf, smi_conf)
        d_model = self.smi_conf.d_model
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(layer, num_layers)

    def forward(
            self, mut_x: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        """
        mut_x: [b, mut_seqlen]
        smi_tgt: [b, smi_seqlen]
        """
        mut_out = self.mut_encoder(mut_x).unsqueeze(-2)  # [b, 1, dmodel]
        smi_out = self.smi_encoder.forward_feature(smi_tgt).unsqueeze(-2)  # [b, 1, dmodel]
        return self.cross_attn(smi_out, mut_out).squeeze(-2)  # [b, dmodel]


class DrugcellAdamr2MutSmisAdd(_DrugcellAdamr2):
    def _forward_mut(self, mut_x: torch.Tensor) -> torch.Tensor:
        """
        mut_x: [b, mut_seqlen]
        out: [b, 1, dmodel]
        """
        return self.mut_encoder(mut_x).unsqueeze(-2)

    def _forward_smi(self, smi_tgt: torch.Tensor) -> torch.Tensor:
        """
        smi_tgt: [b, n, smi_seqlen]
        out: [b, n, dmodel]
        """
        batched = smi_tgt.dim() == 3
        if batched:
            n = smi_tgt.shape[1]
            smi_tgt = smi_tgt.reshape(-1, smi_tgt.shape[-1])
            out = self.smi_encoder.forward_feature(smi_tgt)
            return out.reshape(-1, n, out.shape[-1])
        return self.smi_encoder.forward_feature(smi_tgt)

    def forward(
            self, mut_x: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        mut_out = self._forward_mut(mut_x)
        smi_out = self._forward_smi(smi_tgt)
        return smi_out + mut_out  # [b, n, dmodel]


class DrugcellAdamr2MutSmisXattn(DrugcellAdamr2MutSmisAdd):
    def __init__(self, mut_conf, smi_conf, nhead: int = 2, num_layers: int = 2) -> None:
        super().__init__(mut_conf, smi_conf)
        d_model = smi_conf.d_model
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(layer, num_layers)

    def forward(
            self, mut_x: torch.Tensor, smi_tgt: torch.Tensor) -> torch.Tensor:
        mut_out = self._forward_mut(mut_x)
        smi_out = self._forward_smi(smi_tgt)
        return self.cross_attn(smi_out, mut_out)  # [b, n, dmodel]


class _VAEOmicsAdamr2(nn.Module):

    def __init__(self, omics_conf, smi_conf) -> None:
        super().__init__()
        self.omics_conf = omics_conf
        self.smi_conf = smi_conf

        self.omics_encoder = VAEOmics(omics_conf)
        self.smi_encoder = AdaMR2(smi_conf)

    def load_ckpt(self, *ckpt_files: str) -> None:
        self.load_state_dict(
            torch.load(ckpt_files[0], map_location=torch.device("cpu"))
        )

    def load_pretrained_ckpt(self, omics_ckpt: str, smi_ckpt: str, freeze_omics: bool = False, freeze_smi: bool = False) -> None:
        self.omics_encoder.load_ckpt(omics_ckpt)
        self.smi_encoder.load_ckpt(smi_ckpt)
        if freeze_smi:
            self.smi_encoder.requires_grad_(False)
        if freeze_omics:
            self.omics_encoder.requires_grad_(False)


class VAEOmicsAdamr2OmicsSmiXattn(_VAEOmicsAdamr2):
    def __init__(self, omics_conf, smi_conf, nhead: int = 2, num_layers: int = 2) -> None:
        super().__init__(omics_conf, smi_conf)
        d_model = self.smi_conf.d_model
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(layer, num_layers)

    def forward(
            self, omics: dict, omics_x: typing.Sequence[torch.Tensor], smi_tgt: torch.Tensor) -> torch.Tensor:
        """
        omics_x: [b, omics_seqlen]
        smi_tgt: [b, smi_seqlen]
        """
        omics_out = self.omics_encoder.forward_encoder(omics, *omics_x).unsqueeze(-2)  # [b, 1, dmodel]
        smi_out = self.smi_encoder.forward_feature(smi_tgt).unsqueeze(-2)  # [b, 1, dmodel]
        return self.cross_attn(smi_out, omics_out).squeeze(-2)  # [b, dmodel]


class VAEOmicsAdamr2OmicsSmisAdd(_VAEOmicsAdamr2):
    def _forward_omics(self, omics: dict, omics_x: typing.Sequence[torch.Tensor]) -> torch.Tensor:
        """
        omics_x: [b, omics_seqlen]
        out: [b, 1, dmodel]
        """
        return self.omics_encoder.forward_encoder(omics, *omics_x).unsqueeze(-2)

    def _forward_smi(self, smi_tgt: torch.Tensor) -> torch.Tensor:
        """
        smi_tgt: [b, n, smi_seqlen]
        out: [b, n, dmodel]
        """
        batched = smi_tgt.dim() == 3
        if batched:
            n = smi_tgt.shape[1]
            smi_tgt = smi_tgt.reshape(-1, smi_tgt.shape[-1])
            out = self.smi_encoder.forward_feature(smi_tgt)
            return out.reshape(-1, n, out.shape[-1])
        return self.smi_encoder.forward_feature(smi_tgt)

    def forward(
            self, omics: dict, omics_x: typing.Sequence[torch.Tensor], smi_tgt: torch.Tensor) -> torch.Tensor:
        omics_out = self._forward_omics(omics, omics_x)
        smi_out = self._forward_smi(smi_tgt)
        return smi_out + omics_out  # [b, n, dmodel]


class VAEOmicsAdamr2OmicsSmisXattn(VAEOmicsAdamr2OmicsSmisAdd):
    def __init__(self, omics_conf, smi_conf, nhead: int = 2, num_layers: int = 2) -> None:
        super().__init__(omics_conf, smi_conf)
        d_model = smi_conf.d_model
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.cross_attn = nn.TransformerDecoder(layer, num_layers)

    def forward(
            self, omics: dict, omics_x: typing.Sequence[torch.Tensor], smi_tgt: torch.Tensor) -> torch.Tensor:
        omics_out = self._forward_omics(omics, omics_x)
        smi_out = self._forward_smi(smi_tgt)
        return self.cross_attn(smi_out, omics_out)  # [b, n, dmodel]
