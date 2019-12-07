import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization


# TODO: maybe move code from zonotope.py over here, instead of calling Zonotope methods (seems a bit more natural -- though both work)

class _zModule(nn.Module):
    """
    Attributes:
        in_dim (torch.Size): dimensions of the layer, i.e dimension of the ambient space of the input zonotope
        __out_dim (torch.Size): cache for the layer's output dimension
    """

    def __init__(self):
        super().__init__()
        self.in_dim = None  # "not yet initialized"
        self.__out_dim = None  # "not yet cached"

    def __str__(self):
        return super().__str__().split("(")[0] \
               + " in_dim={} out_dim={}".format(list(self.in_dim), list(self.out_dim()))

    def out_dim(self):
        if self.__out_dim is None:
            self.__out_dim = self._get_out_dim()
        return self.__out_dim

    # this is intended to be overridden by subclasses, so make it "semiprivate" (no name-mangling)
    def _get_out_dim(self):
        raise NotImplementedError("Please implement this method")


# The only _zModule that has parameters
class zReLU(_zModule):
    """
    Applies the ReLU transformation to the zonotope, using self.lambda_layer.
    If self.lambda_layer.requires_grad==False, i.e we're not optimizing over this zlayer's parameters,
        use vanilla DeepZ to update the lambdas (instead of freezing their values, which is clearly not what we want)

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

    def forward(self, zonotope, verbose=False):
        if self.__uninitialized or self.lambda_layer.requires_grad == False:  # if we're not optimizing over this zlayer's parameters, use DeepZ
            self.__uninitialized = False
            if verbose:
                print("zReLU: setting the lambdas for this layer using DeepZ")
            self.__set_lambdas_to_deepz_(zonotope)
        return zonotope.relu(self.lambda_layer)

    # follow pyTorch convention: "__x" prefix is for privateness, "y_" suffix is for in-placeness
    def __set_lambdas_to_deepz_(self, zonotope):
        """Initialize self.lambda_layer to the vanilla DeepZ coefficients on this particular input zonotope"""
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

    def forward(self, zonotope, verbose=False):
        return zonotope.normalization(self.mean, self.sigma)


class zFlatten(_zModule):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

    def _get_out_dim(self):
        return torch.Size([self.in_dim.numel()])

    def forward(self, zonotope, verbose=False):
        return zonotope.flatten()


class zLinear(_zModule):
    """For linear layers, the constructor expects the concrete layer directly.
    Args:
        concrete_layer (nn.Linear): the corresponding layer in the concrete
    """
    def __init__(self, concrete_layer):
        super().__init__()
        self.__concrete_layer = concrete_layer
        self.weight = concrete_layer.weight.detach()
        self.bias = concrete_layer.bias.detach()

    def _get_out_dim(self):
        # take advantage of this call to make a sanity-check on in_dim
        assert len(self.in_dim) == 1
        assert self.in_dim[0] == self.weight.shape[1]
        return self.bias.shape

    def forward(self, zonotope, verbose=False):
        return zonotope.linear_transformation(self.__concrete_layer)


class zConv2d(_zModule):
    """For convolution layers, the constructor expects the concrete layer directly.
    Args:
        conv_layer (nn.Conv2d): the corresponding layer in the concrete
    """
    def __init__(self, concrete_layer):
        super().__init__()
        self.__concrete_layer = concrete_layer
        self.weight = concrete_layer.weight.detach()
        self.bias = concrete_layer.bias.detach()
        if concrete_layer.padding_mode != 'zeros':
            raise UserWarning(
                "There is a Conv2d layer with padding_mode != 'zeros', which is not supported by our analyzer (!!!)")

    def _get_out_dim(self):
        # using the formula given in https://pytorch.org/docs/stable/nn.html#conv2d (section "Shape")
        def make_tuple(x):
            if isinstance(x, int):
                return (x, x)
            else:
                return x
        # torch.nn.Conv2d.kernel_size, stride, padding, dilation are int || tuple of (int, int). out_channels is int.
        kernel_size = make_tuple(self.__concrete_layer.kernel_size)
        stride = make_tuple(self.__concrete_layer.stride)
        padding = make_tuple(self.__concrete_layer.padding)
        dilation = make_tuple(self.__concrete_layer.dilation)
        out_channels = self.__concrete_layer.out_channels
        h_out = (self.in_dim[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        w_out = (self.in_dim[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        return torch.Size([out_channels, h_out, w_out])
        # dummy_input = torch.zeros(1, *self.in_dim)
        # dummy_output = self.forward(Zonotope(dummy_input, dummy_input)) # 0 mean, 1 error term with coefficients 0
        # return dummy_output.dim

    def forward(self, zonotope, verbose=False):
        return zonotope.convolution(self.__concrete_layer)
