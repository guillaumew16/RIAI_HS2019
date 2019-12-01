import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization

class _zModule(nn.Module):
    """
    Attributes:
        in_dim (torch.Size)
        out_dim (torch.Size)
    """
    def __init__():
        self.in_dim = None # means "any"
        self.out_dim = None

    def __str__():
        raise NotImplemented

    @property
    def in_dim():
        raise NotImplemented

    @property
    def out_dim():
        raise NotImplemented

# The only _zModule that has parameters
class zReLU(_zModule):
    """
    Args:
        in_dim: dimensions of the layer, i.e dimension of the ambiant space of the input zonotope
    """
    def __init__(self, in_dim):
        super().__init__()
        self.lambda_layer = Parameter(torch.Tensor(1, *in_dim))
        self.reset_parameters()

    def forward(self, zonotope):
        # TODO: move code over here
        zonotope.relu(self.lambda_layer)

class zNormalization(_zModule):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
    def __init__(self, concrete_layer):
        super().__init__()
        if not isinstance(concrete_layer, Normalization):
            raise ValueError("expected a networks.Normalization")
        self.mean = concrete_layer.mean
        self.mean = concrete_layer.sigma

    def forward(self, zonotope):
        zonotope.normalization()


class zFlatten(_zModule):
    def __init__(self):
        super().__init__()
    def forward(self, zonotope):
        zonotope.flatten()

class zLinear(_zModule):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias
    def __init__(self, concrete_layer):
        super().__init__()
        if not isinstance(concrete_layer, nn.Linear):
            raise ValueError("expected a nn.Linear")
        self.weight = concrete_layer.weight
        self.bias = concrete_layer.bias

    def forward(self, zonotope):
        zonotope.linear_transformation(self.weight, self.bias)

class zConv2d(_zModule):
    def __init__(self, weight, bias, stride, padding, dilation, groups):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        raise NotImplementedError

    def __init__(self, concrete_layer):
        """
        Args:
            conv_layer (nn.Conv2d): the corresponding layer in the concrete
        """
        super().__init__()
        self.weight = concrete_layer.weight
        self.bias   = concrete_layer.bias
        self.stride = concrete_layer.stride
        self.padding = concrete_layer.padding
        self.dilation = concrete_layer.dilation
        self.groups = concrete_layer.groups
        
        self.__concrete_layer = concrete_layer # TODO: once possible, remove this

    def forward(self, zonotope):
        zonotope.convolution(self.__concrete_layer)
