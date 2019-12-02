import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization

import zmodules as zm

# does NOT implemented zm._zModule, but implements nn.Module directly
class zNet(nn.Module):
    """
    Network where the variables are zonotopes and the parameters are lambdas.

    Attributes:
        zlayers (list of zm._zModule): list of layers corresponding to zonotope transformations.
            Each element corresponds to a layer in `self.__net`.
            Note that `self.zlayers` is a Python list, NOT something produced by `nn.Sequential()`, contrary to the `layers` attribute in networks.py.
        zonotopes (list of Zonotope): the list of the intermediary zonotopes (one zonotope for each layer).
            Each element is a Zonotope of shape [1, <*shape of nn layer>].
            `self.zonotopes[i]` is the input of `self.zlayers[i]`. Thus, `self.zonotopes[-1]` is the output zonotope, and `len(self.zonotopes) = len(self.zlayers) + 1`.
        lambdas (list of torch.nn.Parameter): the list of the Znet's parameters (one `lambda_layer` tensor for each ReLU layer).
            Each element `lambda_layer` is a Tensor of shape [1, <*shape of nn layer>].
        relu_ind (list of int): the ordered list of indices i s.t `self.zlayers[i]` is a ReLU layer.
            Thus, `self.lambdas[i]` corresponds to the layer `self.zlayers[i]`.
        __net (networks.FullyConnected || networks.Conv): the corresponding network in the concrete, kept for convenience.

    Args:
        net (networks.FullyConnected || networks.Conv): the corresponding network in the concrete
    """

    def __init__(self, net):
        super().__init__()
        self.__net = net
        for p in net.parameters():
            p.requires_grad = False  # avoid doing useless computations
        
        self.zlayers = []
        out_dim = torch.Size([1, 28, 28]) # the in/out_dim of the **previous** layer
        for layer in net.layers:
            # print(layer, out_dim)
            if isinstance(layer, nn.ReLU):
                zlayer = zm.zReLU(in_dim=out_dim) # needs to set in_dim right away.
            elif isinstance(layer, nn.Linear):
                zlayer = zm.zLinear(layer.weight, layer.bias)
            elif isinstance(layer, nn.Conv2d):
                zlayer = zm.zConv2d(layer)
                # zlayer = zm.zConv2d(
                #     weight=layer.weight, 
                #     bias=layer.bias,
                #     stride=layer.stride,
                #     padding=layer.padding,
                #     dilation=layer.dilation,
                #     groups=layer.groups
                # )
            elif isinstance(layer, Normalization):
                zlayer = zm.zNormalization(layer.mean, layer.sigma)
            elif isinstance(layer, nn.Flatten):
                zlayer = zm.zFlatten()
            zlayer.in_dim = out_dim
            out_dim = zlayer.out_dim()
            self.zlayers.append(zlayer)

        # sanity-check the dimensions (i.e check that pyTorch doesn't do black-magic-broadcasting that ends up being compatible but not what we want)
        # Rk: this is not necessary, it's just a cool upside of the fact that we need to store in_dim for each zlayer
        out_dim = torch.Size([1, 28, 28]) # the out_dim of the **previous** layer
        for zlayer in self.zlayers:
            # print(zlayer)
            assert zlayer.in_dim == out_dim
            out_dim = zlayer.out_dim()
        assert out_dim == torch.Size([10])

        self.zonotopes = [None] * (len(self.zlayers) + 1)

        self.lambdas = []
        self.relu_ind = []
        for idx, zlayer in enumerate(self.zlayers):
            if isinstance(zlayer, zm.zReLU):
                self.lambdas.append(zlayer.lambda_layer) # captures the nn.Parameters (in python, "Object references are passed by value")
                self.relu_ind.append(idx)

    def __str__(self):
        toprint = ""
        for idx, zlayer in enumerate(self.zlayers):
            toprint += "{} {}\n".format(idx, zlayer)
        return toprint.rstrip()

    def forward_step(self, zonotope, zlayer, verbose=False):
        """Applies `layer` to `zonotope` (using `self.lambdas` if `layer` is a ReLU) and returns the result.
        Args:
            inp_zono (Zonotope): the input zonotope
            zlayer (zm._zModule): the layer to apply
        """
        # DEBUG
        if not isinstance(zlayer, zm._zModule):
            raise ValueError("expected a zm._zModule")
        if verbose:
            print("\tapplying zNet.forward_step() at layer: {}\n\ton zonotope: {}".format(zlayer, zonotope))
        return zlayer(zonotope)

    def forward(self, input_zonotope, verbose=False):
        """Run the network transformations on `input_zonotope`.
        Args:
            input_zonotope (Zonotope): the zonotope, of shape A.shape=[784, 1, 28, 28]
        """
        if verbose: print("entering zNet.forward()...")
        zonotope = input_zonotope.reset() # avoid capturing, just in case. (DEBUG)
        self.zonotopes[0] = zonotope
        for idx, zlayer in enumerate(self.zlayers):
            if verbose:
                print("calling zNet.forward_step() on layer #", idx)
                # print("self.zonotopes[{}]: {}".format(idx, self.zonotopes[idx]) )
            self.zonotopes[idx+1] = self.forward_step(self.zonotopes[idx], zlayer, verbose=verbose)
        if verbose: print("finished running zNet.forward().")
        return self.zonotopes[-1]
