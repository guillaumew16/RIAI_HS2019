import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization

class _zModule(nn.Module):
    """
    Attributes:
        in_dim (torch.Size): dimensions of the layer, i.e dimension of the ambiant space of the input zonotope
    """
    def __init__(self):
        super().__init__()
        self.in_dim = None # can be interpreted as "any"...

    def __str__(self):
        return super().__str__() + " in_dim=" + str(self.in_dim) + " out_dim=" + str(self.out_dim())

    def out_dim(self):
        raise NotImplementedError("Please implement this method")

# The only _zModule that has parameters
class zReLU(_zModule):
    """
    Attributes:
        lambda_layer (nn.Parameter): lambda layer of shape [1, *in_dim] (same as zonotopes' a0.shape)
        __uninitialized (bool): True iff `self.lambda_layer` still holds some dummy values, i.e was not yet initialized to the vanilla DeepZ coefficients
    Args:
        in_dim (torch.Size): dimensions of the layer, i.e dimension of the ambiant space of the input zonotope (cf _zModule attributes)
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.__uninitialized = True
        self.lambda_layer = nn.Parameter(torch.zeros(1, *in_dim), requires_grad=True)
    
    def out_dim(self):
        return self.in_dim

    def forward(self, zonotope):
        if self.__uninitialized():
            self.initialize_parameters(zonotope)
        # TODO: move code over here
        zonotope.relu(self.lambda_layer)

    def initialize_parameters(self, zonotope):
        """Initialize self.lambda_layer to the vanilla DeepZ coefficients on this particular input zonotope"""
        self.__uninitialized = False
        self.lambda_layer = zonotope.compute_lambda_breaking_point()

class zNormalization(_zModule):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = mean
        self.sigma = sigma

    # Python doesn't support __init__ overload
    # def __init__(self, concrete_layer):
    #     super().__init__()
    #     if not isinstance(concrete_layer, Normalization):
    #         raise ValueError("expected a networks.Normalization")
    #     self.mean = concrete_layer.mean
    #     self.mean = concrete_layer.sigma

    def out_dim(self):
        return self.in_dim

    def forward(self, zonotope):
        zonotope.normalization(self.mean, self.sigma)

class zFlatten(_zModule):
    def __init__(self):
        super().__init__()
    def out_dim(self):
        if self.in_dim is None:
            raise Warning("Attribute self.in_dim should have been initialized, but is still =None. returning out_dim=None")
            return None
        return torch.Size([ torch.empty(self.in_dim).numel() ]) # TODO: check whether torch.Size supports a better way to do this...
    def forward(self, zonotope):
        zonotope.flatten()

class zLinear(_zModule):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    # Python doesn't support __init__ overload
    # def __init__(self, concrete_layer):
    #     super().__init__()
    #     if not isinstance(concrete_layer, nn.Linear):
    #         raise ValueError("expected a nn.Linear")
    #     self.weight = concrete_layer.weight
    #     self.bias = concrete_layer.bias

    def out_dim(self):
        # take advantage of this call to make a sanity-check on in_dim
        assert len(self.in_dim) == 1
        assert self.in_dim[0] == self.weight.shape[1]
        return self.weight.shape[:1]

    def forward(self, zonotope):
        zonotope.linear_transformation(self.weight, self.bias)

class zConv2d(_zModule):

    # Python doesn't support __init__ overload
    # def __init__(self, weight, bias, stride, padding, dilation, groups):
    #     super().__init__()
    #     self.weight = weight
    #     self.bias = bias
    #     self.stride = stride
    #     self.padding = padding
    #     self.dilation = dilation
    #     self.groups = groups
    #     raise NotImplementedError

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

    def out_dim(self):
        # TODO: maybe try to find a better way to do this... On the other hand this is not supposed be called very often anyway
        dummy_input = torch.zeros(1, *self.in_dim)
        # dummy_output = self.forward(Zonotope(A=dummy_input, a0=dummy_input))
        # return dummy_output.dim
        dummy_output = __concrete_layer(dummy_input) # since it's there...
        return dummy_output.shape[1:]

    def forward(self, zonotope):
        zonotope.convolution(self.__concrete_layer)
