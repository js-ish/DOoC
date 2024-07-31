# DOoC

## Train

```python
import random
import torch
from torch import nn
import torch.optim as optim

from moltx import tokenizers as tkz
from moltx.models import AdaMRTokenizerConfig

from dooc import models, datasets
```

### Regression

```python
# Regression datasets
tokenizer = tkz.MoltxTokenizer.from_pretrain(
    conf=AdaMRTokenizerConfig.Prediction
    )
ds = datasets.MutSmiReg(smi_tokenizer=tokenizer)
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"]
mutations = [[random.choice([0, 1]) for _ in range(3008)]] * 2
# mutations contains 0/1 encoding information of the genome
values = [0.85, 0.78]
mut_x, smi_tgt, out = ds(mutations, smiles, values)

# Regression train
model = models.MutSmiReg()
model.load_pretrained_ckpt(
    mut_ckpt='path/to/drugcell.pt',
    smi_ckpt='path/to/moltx.ckpt'
    )
mse_loss = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-04,
    foreach=False
    )
optimizer.zero_grad()
pred = model(mut_x=mut_x, smi_tgt=smi_tgt)
loss = mse_loss(pred, out)
loss.backward()
optimizer.step()

torch.save(model.state_dict(), '/path/to/mutsmireg.ckpt')
```

### Pairwise

```python
# Pairwise datasets
tokenizer = tkz.MoltxTokenizer.from_pretrain(
    conf=AdaMRTokenizerConfig.Prediction
    )
ds = datasets.MutSmisPairwiseRank(smi_tokenizer=tokenizer)
smiles = [["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br"],
          ["CC[N+](C)(C)Cc1ccccc1Br", "CN(Cc1oc2ccccc2c1C)C(=O)\C=C\c1cnc2NC(=O)CCc2c1"]]
mutations = [[random.choice([0, 1]) for _ in range(3008)]] * 2
# mutations contains 0/1 encoding information of the genome
values = [[0.85, 0.78]] * 2
mut_x, smi_tgt, out = ds(mutations, smiles, values)

# Pairwise train
model = models.MutSmisRank()
model.load_pretrained_ckpt(
    mut_ckpt='path/to/drugcell.pt',
    smi_ckpt='path/to/moltx.ckpt'
    )

# Pairwise loss
loss_func = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-04,
    foreach=False
    )

optimizer.zero_grad()
pred = model(mut_x=mut_x, smi_tgt=smi_tgt)
loss = loss_func(pred[:,0] - pred[:,1], out)
loss.backward()
optimizer.step()
torch.save(model.state_dict(), '/path/to/mutsmipairwise.ckpt')
```

### Listwise

```python
# Listwise datasets
tokenizer = tkz.MoltxTokenizer.from_pretrain(
    conf=AdaMRTokenizerConfig.Prediction
    )
ds = datasets.MutSmisListwiseRank(smi_tokenizer=tokenizer)
smiles = [["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br", "CN(Cc1oc2ccccc2c1C)C(=O)\C=C\c1cnc2NC(=O)CCc2c1"]] * 2
mutations = [[random.choice([0, 1]) for _ in range(3008)]] * 2
# mutations contains 0/1 encoding information of the genome
values = [[0.85, 0.78, 0.79]] * 2
mut_x, smi_tgt, out = ds(mutations, smiles, values)
mut_x, smi_tgt, out = mut_x.squeeze(0), smi_tgt.squeeze(0), out.squeeze(0)

# Listwise train
model = models.MutSmisRank()
model.load_pretrained_ckpt(
    mut_ckpt='path/to/drugcell.pt',
    smi_ckpt='path/to/moltx.ckpt'
    )

# Listwise loss
loss_func = dooc_list_loss.ListNetLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-04,
    foreach=False
    )

optimizer.zero_grad()
pred = model(mut_x=mut_x, smi_tgt=smi_tgt)
loss = loss_func(pred, out)
loss.backward()
optimizer.step()

torch.save(model.state_dict(), '/path/to/mutsmilistwise.ckpt')
```

## Inference

```python
import random
from moltx import tokenizers as tkz
from moltx.models import AdaMRTokenizerConfig
from dooc import pipelines, models

# Regression
tokenizer = tkz.MoltxTokenizer.from_pretrain(
    conf=AdaMRTokenizerConfig.Prediction
    )
model = models.MutSmiReg()
model.load_ckpt('/path/to/mutsmireg.ckpt')
pipeline = pipelines.MutSmiReg(
    smi_tokenizer=tokenizer, model=model
    )
mutations = [random.choice([0, 1]) for _ in range(3008)]
smiles = "CC[N+](C)(C)Cc1ccccc1Br"
predict = pipeline(mut=mutations, smi=smiles) # e.g. 0.85

# Rank
tokenizer = tkz.MoltxTokenizer.from_pretrain(
    conf=AdaMRTokenizerConfig.Prediction
    )
model = models.MutSmisRank()
model.load_ckpt('/path/to/mutsmirank.ckpt')
pipeline = pipelines.MutSmisRank(smi_tokenizer=tokenizer, model=model)
mutations = [random.choice([0, 1]) for _ in range(3008)]
smiles = ["c1cccc1c", "CC[N+](C)(C)Cc1ccccc1Br", "CN(Cc1oc2ccccc2c1C)C(=O)\C=C\c1cnc2NC(=O)CCc2c1"]
predict = pipeline(mut=mutations, smis=smiles) # e.g. ["CN(Cc1oc2ccccc2c1C)C(=O)\C=C\c1cnc2NC(=O)CCc2c1", "CC[N+](C)(C)Cc1ccccc1Br", "c1cccc1c"]
```
