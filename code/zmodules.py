import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization

# TODO: maybe move code from zonotope.py over here, instead of calling Zonotope methods (seems a bit more natural -- though both work)

class _zModule(nn.Module):
    """
    Attributes:
        in_dim (torch.Size): dimensions of the layer, i.e dimension of the ambient space of the input zonotope
    """
    def __init__(self):
        super().__init__()
        self.in_dim = None # can be interpreted as "any"...
        self.__out_dim = None

    def __str__(self):
        return super().__str__() + " in_dim={} out_dim={}".format(self.in_dim, self.out_dim())

    def out_dim(self):
        if self.__out_dim is None:
            self.__out_dim = self._get_out_dim()
        return self.__out_dim
    
    # this is intended to be overridden by subclasses, so make it "semiprivate"
    def _get_out_dim(self):
        raise NotImplementedError("Please implement this method")

# The only _zModule that has parameters
class zReLU(_zModule):
    """
    Attributes:
        lambda_layer (nn.Parameter): lambda layer of shape [1, *in_dim] (same as zonotopes' a0.shape)
        __uninitialized (bool): True iff `self.lambda_layer` still holds some dummy values, i.e was not yet initialized to the vanilla DeepZ coefficients
    Args:
        in_dim (torch.Size): dimensions of the layer, i.e dimension of the ambient space of the input zonotope (cf _zModule attributes)
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.__uninitialized = True
        self.lambda_layer = nn.Parameter(torch.zeros(1, *in_dim), requires_grad=True)
    
    def _get_out_dim(self):
        return self.in_dim

    def forward(self, zonotope):
        if self.__uninitialized:
            self.initialize_parameters(zonotope)
        return zonotope.relu(self.lambda_layer)

    def initialize_parameters(self, zonotope):
        """Initialize self.lambda_layer to the vanilla DeepZ coefficients on this particular input zonotope"""
        self.__uninitialized = False
        # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
        # https://pytorch.org/docs/master/_modules/torch/nn/init.html
        with torch.no_grad():
            self.lambda_layer.data = zonotope.compute_lambda_breaking_point()

class zNormalization(_zModule):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
    def _get_out_dim(self):
        return self.in_dim
    def forward(self, zonotope):
        return zonotope.normalization(self.mean, self.sigma)

class zFlatten(_zModule):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
    def _get_out_dim(self):
        return torch.Size([ torch.empty(self.in_dim).numel() ]) # TODO: check whether torch.Size supports a better way to do this...
    def forward(self, zonotope):
        return zonotope.flatten()

class zLinear(_zModule):
    def __init__(self, layer):
        super().__init__()
        self.weight = weight.detach()
        self.bias = bias.detach()
    def _get_out_dim(self):
        # take advantage of this call to make a sanity-check on in_dim. for DEBUG only
        assert len(self.in_dim) == 1
        assert self.in_dim[0] == self.weight.shape[1]
        return self.weight.shape[:1]
    def forward(self, zonotope):
        return zonotope.linear_transformation(self.weight, self.bias)

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
        self.weight = concrete_layer.weight.detach()
        self.bias   = concrete_layer.bias.detach()
        self.stride = concrete_layer.stride # stride is just a tuple of ints
        self.padding = concrete_layer.padding # padding too
        self.dilation = concrete_layer.dilation # dilation too
        self.groups = concrete_layer.groups # groups is just an int

    def _get_out_dim(self):
        # TODO: maybe try to find a better way to do this... 
        # On the other hand multiplying by 0 is probably optimized out by pyTorch, and anyway the result is cached.
        dummy_input = torch.zeros(1, *self.in_dim)
        dummy_output = self.forward(Zonotope(A=dummy_input, a0=dummy_input))
        self.__out_dim = dummy_output.dim
        return self.__out_dim

    def forward(self, zonotope):
        return zonotope.convolution(self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
