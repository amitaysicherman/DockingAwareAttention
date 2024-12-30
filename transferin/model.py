import os
from enum import Enum
import dataclasses
import torch
from torch import nn as nn
from torch.nn import functional as F


class DataType(Enum):
    MOLECULE = 'M'
    PROTEIN = 'P'


@dataclasses.dataclass
class ReactEmbedConfig:
    p_dim: int
    m_dim: int
    n_layers: int
    hidden_dim: int
    dropout: float
    normalize_last: int = 1


def get_layers(dims, dropout=0.0):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        if dropout > 0:
            layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
    return layers


class ReactEmbedModel(nn.Module):
    def __init__(self, config: ReactEmbedConfig):
        super(ReactEmbedModel, self).__init__()
        self.config = config
        if config.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.layers_dict = nn.ModuleDict()
        for src_dim, src in zip([config.p_dim, config.m_dim], ["P", "M"]):
            for dst_dim, dst in zip([config.p_dim, config.m_dim], ["P", "M"]):
                name = f"{src}-{dst}"
                dims = [src_dim] + [config.hidden_dim] * (config.n_layers - 1) + [dst_dim]
                self.layers_dict[name] = get_layers(dims, config.dropout)

    def forward(self, x, type_):
        x = self.layers_dict[type_](x)
        if self.config.normalize_last:
            return F.normalize(x, dim=-1)
        else:
            return x

    def dual_forward(self, x, type_1):
        if type_1 == "P":
            y1 = self.forward(x, "P-P")
            y2 = self.forward(x, "P-M")
        else:
            y1 = self.forward(x, "M-P")
            y2 = self.forward(x, "M-M")
        y = torch.cat([y1, y2], dim=-1)
        return y


def get_layers2(dims, dropout):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
    return layers


class LinFuseModel(nn.Module):
    def __init__(self, dtype_1: DataType, output_dim=1, fuse_model=None, n_layers=2, drop_out=0.3, hidden_dim=512):
        super().__init__()
        self.use_fuse = True
        self.use_model = False
        self.fuse_model = fuse_model
        self.fuse_dim = fuse_model.config.p_dim + fuse_model.config.m_dim
        self.input_dim = self.fuse_dim
        self.dtype = dtype_1
        hidden_layers = [hidden_dim] * (n_layers - 1)
        self.layers = get_layers2([self.input_dim] + hidden_layers + [output_dim], dropout=drop_out)

    def forward(self, data):
        x = self.fuse_model.dual_forward(data, self.dtype.value).detach()
        return self.layers(x)


def get_model(cp_path="transferin/model.pt"):
    config = ReactEmbedConfig(p_dim=1152, m_dim=768, n_layers=1, hidden_dim=256, dropout=0.3)
    model = ReactEmbedModel(config)
    task_model = LinFuseModel(DataType.PROTEIN, fuse_model=model)
    if cp_path is not None:
        task_model.load_state_dict(torch.load(cp_path, map_location="cpu"))
    return task_model


if __name__ == "__main__":
    model = get_model()
    print(model)
    x = torch.rand(10, 1152)
    y = model(x)
    print(y.shape)
    y_fuse = model.fuse_model.dual_forward(x, "P")
    print(y_fuse.shape)
    # model = get_model()
    # print(model)
    # x = torch.rand(10, 1152 + 768)
    # y = model(x)
    # print(y)
    # print(y.shape)
