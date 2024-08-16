import random
from dooc import pipelines, models


def test_MutSmiReg(smi_tkz):
    mutation = [random.choice([1, 0]) for _ in range(3008)]
    smiles = "CC[N+](C)(C)Cc1ccccc1Br"

    model = models.MutSmiReg()
    pipeline = pipelines.MutSmiReg(smi_tokenizer=smi_tkz,
                   model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, float)


def test_MutSmisRank(smi_tkz):
    mutation = [random.choice([1, 0]) for _ in range(3008)]
    smiles = ["CC[N+](C)(C)Cc1ccccc1Br", "CC[N+](C)(C)Cc1ccccc1Br", "c1cccc1c"]

    class Pointwise(pipelines._MutSmi, pipelines._MutSmisRank):
        pass

    model = models.MutSmiReg()
    pipeline = Pointwise(smi_tokenizer=smi_tkz, model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] == "CC[N+](C)(C)Cc1ccccc1Br"

    model = models.MutSmisRank()
    pipeline = pipelines.MutSmisRank(smi_tokenizer=smi_tkz, model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] == "CC[N+](C)(C)Cc1ccccc1Br"

    model = models.MutSmisRankV2()
    pipeline = pipelines.MutSmisRank(smi_tokenizer=smi_tkz, model=model)
    out = pipeline(mutation, smiles)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] == "CC[N+](C)(C)Cc1ccccc1Br"


def test_MultiOmicsSmisRank(smi_tkz):
    mutation = [random.choice([1, 0]) for _ in range(3008)]
    rna = [random.random() for _ in range(5537)]
    pathway = [random.random() for _ in range(3793)]
    smiles = ["CC[N+](C)(C)Cc1ccccc1Br", "CC[N+](C)(C)Cc1ccccc1Br", "c1cccc1c"]

    class Pointwise(pipelines._MultiOmicsSmis, pipelines._MultiOmicsSmisRank):
        pass

    model = models.MultiOmicsSmisRank()
    pipeline = pipelines.MultiOmicsSmisRank(smi_tokenizer=smi_tkz, model=model)
    out = pipeline(mutation, rna, pathway, smiles)
    assert isinstance(out, list)
    assert len(out) == 3
    assert out[1] == "CC[N+](C)(C)Cc1ccccc1Br"
