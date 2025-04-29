# tests/test_train_script.py

import yaml
import numpy as np
import torch
import pytest
from torch.utils.data import TensorDataset, DataLoader

from scripts.train import load_config, TokenDataset, train_one_epoch, evaluate

def test_load_config(tmp_path):
    # write out a minimal config
    cfg = {
        "block_size": 4,
        "batch_size": 2,
        "vocab_size": 10,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "epochs": 1
    }
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump(cfg))
    loaded = load_config(str(cfg_file))
    assert isinstance(loaded, dict)
    for k, v in cfg.items():
        assert loaded[k] == v

def test_tokendataset_len_and_getitem(tmp_path):
    # create a tiny .bin file with uint16 data [0,1,2,3,4,5]
    arr = np.arange(6, dtype=np.uint16)
    path = tmp_path / "data.bin"
    arr.tofile(str(path))

    ds = TokenDataset(str(path), block_size=2)
    # length should be len(arr) - block_size
    assert len(ds) == 6 - 2

    # __getitem__(1) should yield x=[1,2], y=[2,3]
    x, y = ds[1]
    assert torch.equal(x, torch.tensor([1, 2], dtype=torch.long))
    assert torch.equal(y, torch.tensor([2, 3], dtype=torch.long))

class DummyModel(torch.nn.Module):
    """
    A tiny model with 1 parameter so that loss.backward() and optimizer.step()
    actually run without error.
    """
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x, y):
        # pretend prediction = x * param
        x = x.float()
        y = y.float()
        pred = x * self.param
        # simple MSE loss
        loss = torch.mean((pred - y) ** 2)
        return pred, loss

@pytest.fixture
def toy_loader():
    # 10 samples of shape (3,), x=1, y=0
    x = torch.ones((10, 3), dtype=torch.long)
    y = torch.zeros((10, 3), dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=5)

def test_train_one_epoch_and_evaluate(toy_loader):
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    device = torch.device("cpu")
    epoch = 0

    # train should run without error and return a float
    train_loss = train_one_epoch(model, toy_loader, optimizer, device, epoch)
    assert isinstance(train_loss, float)
    assert train_loss >= 0

    # evaluation should also run and return a float
    eval_loss = evaluate(model, toy_loader, device)
    assert isinstance(eval_loss, float)
    assert eval_loss >= 0
