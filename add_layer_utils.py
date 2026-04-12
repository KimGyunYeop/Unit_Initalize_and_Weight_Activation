import math

import torch
from torch import nn


def _build_weight_activation(act_type):
    if act_type == "midrelu":
        return nn.ReLU()
    if act_type == "leakyrelu":
        return nn.LeakyReLU()
    if act_type == "midgelu":
        return nn.GELU()
    raise ValueError("invalid weight activation type: {}".format(act_type))


def _init_bias_if_needed(module):
    if getattr(module, "bias", None) is None:
        return
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(module.bias, -bound, bound)


def initialize_added_layer(module, init_type="unit"):
    if init_type not in ["unit", "xavier", "he", "zero"]:
        raise ValueError("invalid initialize type of adding layer: {}".format(init_type))

    if module.weight.size(0) != module.weight.size(1):
        raise ValueError("in current version, in feature and out feature must be same")

    if init_type == "unit":
        module.weight.data.copy_(torch.eye(module.weight.size(0), device=module.weight.device, dtype=module.weight.dtype))
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        return module

    if init_type == "xavier":
        nn.init.xavier_uniform_(module.weight)
        _init_bias_if_needed(module)
        return module

    if init_type == "he":
        nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity="relu")
        _init_bias_if_needed(module)
        return module

    nn.init.zeros_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)
    return module


class CustomLinearLayer(nn.Module):
    """Custom linear layer that applies activation to the weights before matmul."""

    def __init__(self, size_in, size_out, bias=False, act_type="midgelu", init_type="unit"):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.weight_act = _build_weight_activation(act_type)

        self.weight = nn.Parameter(torch.empty(size_out, size_in))
        self.bias = nn.Parameter(torch.empty(size_out)) if bias else None

        initialize_added_layer(self, init_type=init_type)

    def forward(self, x):
        x = torch.matmul(x, self.weight_act(self.weight).t())
        if self.bias is not None:
            x = torch.add(x, self.bias)
        return x


def add_unit_init_linear(in_features, out_features, bias=False, init_type="unit", act_type=None):
    if in_features != out_features:
        raise ValueError("in current version, in feature and out feature must same")

    if act_type is None:
        new_layer = nn.Linear(in_features, out_features, bias=bias)
    elif "mid" in act_type or act_type == "leakyrelu":
        new_layer = CustomLinearLayer(
            in_features,
            out_features,
            bias=bias,
            act_type=act_type,
            init_type=init_type,
        )
        return new_layer
    else:
        new_layer = nn.Linear(in_features, out_features, bias=bias)

    initialize_added_layer(new_layer, init_type=init_type)
    return new_layer
