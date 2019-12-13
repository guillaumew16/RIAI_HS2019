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
        zonotopes (list of Zonotope): the list of the intermediary zonotopes (one zonotope for each zlayer + one for the output).
            Each element is a Zonotope of shape [1, <*shape of nn layer>].
            `self.zonotopes[i]` is the input of `self.zlayers[i]`. Thus, `self.zonotopes[-1]` is the output zonotope, and `len(self.zonotopes) = len(self.zlayers) + 1`.
        lambdas (list of torch.nn.Parameter): the list of the Znet's parameters (one `lambda_layer` tensor for each ReLU layer).
            Each element `lambda_layer` is a Tensor of shape [1, <*shape of nn layer>].
        relu_ind (list of int): the ordered list of indices i s.t `self.zlayers[i]` is a ReLU layer.
            Thus, `self.lambdas[i]` corresponds to the layer `self.zlayers[i]`.
        __net (networks.FullyConnected || networks.Conv): the corresponding network in the concrete, kept for convenience.

    Args:
        net (networks.FullyConnected || networks.Conv): the corresponding network in the concrete
        input_shape (torch.Size): the expected shape of a single input of the network, default = (1, 28, 28), which is the shape of a mnist image.
        nb_classes (int, optional): the number of classes for the dataset, default=10, which is the number of classes in mnist.
    """

    def __init__(self, net, input_shape=torch.Size([1, 28, 28]), nb_classes=10):
        super().__init__()
        self.__net = net
        for p in net.parameters():
            p.requires_grad = False  # avoid doing useless computations

        self.zlayers = []
        out_dim = input_shape  # the in/out_dim of the **previous** layer
        for layer in net.layers:
            # print(layer, out_dim)
            if isinstance(layer, nn.ReLU):
                zlayer = zm.zReLU(in_dim=out_dim)  # needs to set in_dim right away.
            elif isinstance(layer, nn.Linear):
                zlayer = zm.zLinear(layer)  # the constructor expects the concrete layer directly.
            elif isinstance(layer, nn.Conv2d):
                zlayer = zm.zConv2d(layer)  # the constructor expects the concrete layer directly.
            elif isinstance(layer, Normalization):
                zlayer = zm.zNormalization(layer.mean, layer.sigma)
            elif isinstance(layer, nn.Flatten):
                zlayer = zm.zFlatten(in_dim=out_dim)  # needs to set in_dim right away.
            zlayer.in_dim = out_dim
            out_dim = zlayer.out_dim()
            self.zlayers.append(zlayer)

        # sanity-check the dimensions (i.e check that pyTorch doesn't do black-magic-broadcasting that ends up being compatible but not what we want)
        # Rk: this is not necessary, it's just a cool upside of the fact that we need to store in_dim for each zlayer
        out_dim = input_shape
        for zlayer in self.zlayers:
            # print(zlayer)
            assert zlayer.in_dim == out_dim
            out_dim = zlayer.out_dim()
        assert out_dim == torch.Size([nb_classes])

        self.zonotopes = [None] * (len(self.zlayers) + 1)

        self.lambdas = []
        self.relu_ind = []
        for idx, zlayer in enumerate(self.zlayers):
            if isinstance(zlayer, zm.zReLU):
                self.lambdas.append(
                    zlayer.lambda_layer)  # captures the nn.Parameters (in python, "Object references are passed by value")
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
        if not isinstance(zlayer, zm._zModule):
            raise ValueError("expected a zm._zModule")
        if verbose:
            print("\tapplying zNet.forward_step() at layer: {}\n\ton zonotope: {}".format(zlayer, zonotope))
        return zlayer(zonotope, verbose)

    def forward(self, input_zonotope, verbose=False):
        """Run the network transformations on `input_zonotope`.
        Args:
            input_zonotope (Zonotope): the zonotope, of shape A.shape=[784, 1, 28, 28]
        """
        if verbose: print("entering zNet.forward()...")
        self.zonotopes[0] = input_zonotope.reset()  # avoid capturing, just in case. (TODO: is this actually useful?)
        for idx, zlayer in enumerate(self.zlayers):
            if verbose:
                print("calling zNet.forward_step() on layer #", idx)
                # print("self.zonotopes[{}]: {}".format(idx, self.zonotopes[idx]) )
            self.zonotopes[idx + 1] = self.forward_step(self.zonotopes[idx], zlayer, verbose=verbose)
        if verbose: print("finished running zNet.forward().")
        return self.zonotopes[-1]


# same as zNet, zLoss (and its subclasses) does NOT implemented zm._zModule, but implements nn.Module directly
class zLoss(nn.Module):
    """
    A wrapper class for all the loss function implementations.
    Takes the logit-layer zonotope as input and should "translate" the property that the (arg)max of the logits is true_label, namely:
        IF self.forward() returns a value <= 0, THEN the property is proved
    """

    def __init__(self):
        super().__init__()

    @property
    def has_lambdas(self):
        raise NotImplementedError("Do not use the class zLoss directly, but one of its subclasses")

    def forward(self, zonotope, verbose=False):
        # fake abstract class
        raise NotImplementedError("Do not use the class zLoss directly, but one of its subclasses")


class zMaxSumOfViolations(zLoss):
    """
    The max sum of violations over the zonotope:
        max_{x(=logit) in output_zonotope} sum_{label l s.t logit[l] > logit[true_label]} (logit[l] - logit[true_label])
    Actually we didn't find any other reasonable choice of loss, cf formulas.pdf.
    Args:
        true_label (int): the true label of the region to verify, with 0 <= true_label < nb_classes.
        nb_classes (int, optional): the number of classes for the dataset, default=10, which is the number of classes in mnist.
    """

    def __init__(self, true_label, nb_classes=10):
        super().__init__()
        assert true_label in range(0, nb_classes)
        self.true_label = true_label
        self.nb_classes = nb_classes
        self.__relu_zlayer = zm.zReLU(in_dim=torch.Size([self.nb_classes]))

    @property
    def has_lambdas(self):
        return True

    @property
    def logit_lambdas(self):
        """Return the lambdas used by the ReLU layer.
        Note that this leaks the object, which is what we want, since we need to pass it as parameter to the optimizer.

        Return: nn.Parameter of shape torch.Size([1, self.nb_classes])
        """
        return self.__relu_zlayer.lambda_layer

    def forward(self, zonotope, verbose=False):
        """
        Args:
            zonotope (Zonotope): the zonotope corresponding to the last layer of a zNet, i.e the zonotope of a logit vector
        """
        assert zonotope.dim == torch.Size([self.nb_classes])
        if verbose:
            print("entering zMaxSumOfViolations.forward()...")
        violation_zono = zonotope - zonotope[self.true_label]
        violation_zono = self.__relu_zlayer(violation_zono, verbose)
        res = violation_zono.sum().upper()
        if verbose:
            print("finished running zMaxSumOfViolations.forward().")
        return res


class zMaxViolation(zLoss):
    """
    The max violation is computed as:
        max_{label l} max_{epsilons} (logit[l] - logit[true_label])
    Args:
        true_label (int): the true label of the region to verify, with 0 <= true_label < nb_classes.
        nb_classes (int, optional): the number of classes for the dataset, default=10, which is the number of classes in mnist.
    """

    def __init__(self, true_label, nb_classes=10):
        super().__init__()
        assert true_label in range(0, nb_classes)
        self.true_label = true_label
        self.nb_classes = nb_classes

    @property
    def has_lambdas(self):
        return False

    def forward(self, zonotope, verbose=False):
        """
        Args:
            zonotope (Zonotope): the zonotope corresponding to the last layer of a zNet, i.e the zonotope of a logit vector
        """
        assert zonotope.dim == torch.Size([self.nb_classes])
        if verbose:
            print("entering zMaxViolation.forward()...")
        violation_zono = zonotope - zonotope[self.true_label]
        res = violation_zono.upper().max()
        if verbose:
            print("finished running zMaxSumOfViolations.forward().")
        return res
