import typing
from functools import cmp_to_key
import torch
import torch.nn as nn
from moltx import tokenizers


class _MutSmiBase:
    def __init__(self, smi_tokenizer: tokenizers.MoltxTokenizer, model: nn.Module, device: torch.device = torch.device("cpu")) -> None:
        self.smi_tokenizer = smi_tokenizer
        self.device = device
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        self.model = model

    def _tokens2tensor(self, tokens: typing.Sequence[int], size: int = None) -> torch.Tensor:
        if size is None:
            size = len(tokens)
        if len(tokens) > size:
            raise IndexError('the length of tokens is greater than size!')
        out = torch.zeros(size, dtype=torch.int)
        for i, tk in enumerate(tokens):
            out[i] = tk
        return out.to(self.device)


class _MutSmi(_MutSmiBase):

    def _model_args(self, mut: typing.Sequence[int], smi: str) -> typing.Tuple[torch.Tensor]:
        mut_x = torch.tensor(mut, device=self.device)
        smi_tgt = self._tokens2tensor(self.smi_tokenizer(self.smi_tokenizer.BOS + smi + self.smi_tokenizer.EOS))
        return mut_x, smi_tgt

    def reg(self, mut: typing.Sequence[int], smi: str) -> float:
        return self.model(*self._model_args(mut, smi)).item()

    def cmp_smis_func(self, mut: typing.Sequence[int]) -> typing.Callable:
        cmped = {}

        def cmp(smi1, smi2):
            query = '-'.join([smi1, smi2])
            if query in cmped:
                return cmped[query]
            out1 = self.reg(mut, smi1)
            out2 = self.reg(mut, smi2)
            out = out1 - out2
            cmped[query] = out
            return out
        return cmp


class _MutSmis(_MutSmiBase):

    def _smi_args(
        self, smis: typing.Sequence[str]
    ) -> torch.Tensor:
        smi_tgt = [self.smi_tokenizer(self.smi_tokenizer.BOS + smi + self.smi_tokenizer.EOS) for smi in smis]
        size_tgt = max(map(len, smi_tgt))
        smi_tgt = torch.concat([self._tokens2tensor(smi, size_tgt).unsqueeze(0) for smi in smi_tgt])
        return smi_tgt

    def cmp_smis_func(self, mut: typing.Sequence[int]) -> typing.Callable:
        mut_x = torch.tensor(mut, device=self.device)
        cmped = {}

        def cmp(smi1, smi2):
            smis = [smi1, smi2]
            query = tuple(smis)
            if query in cmped:
                return cmped[query]
            smi_tgt = self._smi_args(smis)
            out = self.model.forward_cmp(mut_x, smi_tgt)
            cmped[query] = out
            return out
        return cmp


class _MutSmiReg:

    def __call__(self, mut: typing.Sequence[int], smi: str) -> typing.Dict:
        return self.reg(mut, smi)


class _MutSmisRank:

    def __call__(self, mut: typing.Sequence[int], smis: typing.Sequence[str]) -> typing.Sequence[str]:
        return sorted(smis, key=cmp_to_key(self.cmp_smis_func(mut)))


"""
Mutations(Individual Sample) and Smiles Interaction

MutSmiReg
MutSmisRank
MutsSmiRank
"""


class MutSmiReg(_MutSmi, _MutSmiReg):
    pass


class MutSmisRank(_MutSmis, _MutSmisRank):
    def __call__(self, mut: typing.Sequence[int], smis: typing.Sequence[str]) -> typing.Sequence[str]:
        _mut_smis_rank:_MutSmisRank = _MutSmisRank()
        return _mut_smis_rank(mut=mut, smis=smis)
