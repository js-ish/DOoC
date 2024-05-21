import pytest
from dooc import datasets
import random
from moltx import tokenizers as tkz
from moltx.models import AdaMRTokenizerConfig


def test_MutSmi():
    return


def test_MutSmiXAttention():
    tokenizer = tkz.MoltxTokenizer.from_pretrain(
        conf=AdaMRTokenizerConfig.Prediction
        )
    ds = datasets.MutSmiXAttention(tokenizer)
    smiles = ["CC[N+]CCBr", "Cc1ccc1"]
    values = [0.88, 0.89]
    mutations = [[random.choice([0, 1]) for _ in range(3008)],
                 [random.choice([0, 1]) for _ in range(3008)]]
    with pytest.raises(RuntimeError):
        ds(smiles, mutations, values[:1])
    smiles_src, smiles_tgt, mutations_src, out = ds(smiles,
                                                    mutations,
                                                    values)
    assert smiles_src.shape == (2, 200)
    assert smiles_tgt.shape == (2, 200)
    assert mutations_src.shape == (2, 3008)
    assert out.shape == (2, 1)


def test_MutSmiFullConnection():
    tokenizer = tkz.MoltxTokenizer.from_pretrain(
        conf=AdaMRTokenizerConfig.Prediction
        )
    ds = datasets.MutSmiFullConnection(tokenizer)
    smiles = ["CC[N+]CCBr", "Cc1ccc1"]
    values = [0.88, 0.89]
    mutations = [[random.choice([0, 1]) for _ in range(3008)],
                 [random.choice([0, 1]) for _ in range(3008)]]
    with pytest.raises(RuntimeError):
        ds(smiles, mutations, values[:1])
    smiles_src, smiles_tgt, mutations_src, out = ds(smiles,
                                                    mutations,
                                                    values)
    assert smiles_src.shape == (2, 200)
    assert smiles_tgt.shape == (2, 200)
    assert mutations_src.shape == (2, 3008)
    assert out.shape == (2, 1)
