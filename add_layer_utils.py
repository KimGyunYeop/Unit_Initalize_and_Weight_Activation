import torch
import math
from torch import nn

class CustomLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, bias=False, act_type="midgelu"):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        
        weight = torch.Tensor(size_out, size_in)
        # weight = torch.eye(size_out, size_in)
        if act_type == "midrelu":
            self.weight_act = nn.ReLU()
        elif act_type == "leakyrelu":
            self.weight_act = nn.LeakyReLU(negative_slope=1.0/size_in)
        elif act_type == "midgelu":
            self.weight_act = nn.GELU()
        else:
            assert "act type error"

        # print(weights)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter
        self.bias = bias
        if bias:
            bias = torch.Tensor(size_out)
            self.bias = nn.Parameter(bias)
        
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        # bound = 1 / math.sqrt(fan_in)
        # nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x= torch.matmul(x, self.weight_act(self.weight).t())
        
        if self.bias:
            x = torch.add(x, self.bias)
        
        return x  # w times x + b
    

def add_unit_init_linear(in_features, out_features, bias=False, init_type="unit", act_type=None):
    
    if init_type.lower() not in ["unit", "he"]:
        assert "invaild initalize type of adding layer!!"
        
    if in_features != out_features:
        assert "In current version, in feature and out feature must same!!"
    
    
    if act_type is None:
        new_layer = nn.Linear(in_features, out_features, bias=bias)
    elif "mid" in act_type:
        new_layer = CustomLinearLayer(in_features, out_features, bias=bias, act_type=act_type)
    else:
        new_layer = nn.Linear(in_features, out_features, bias=bias)
    
    if init_type == "unit":
        new_layer.weight.data = torch.nn.Parameter(torch.eye(in_features))
        
    # print(new_layer.weight)
        
    
    return new_layer
    