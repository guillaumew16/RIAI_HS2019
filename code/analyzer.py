import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization

from znet import zNet

class Analyzer:
    """
    Analyzer expected by `verifier.py`, to be run using `Analyzer.analyze()`.
    In terms of the attributes, the query to be answered is:
        ?"forall x in input_zonotope, net(x) labels x as true_label"?

    `loss(forward( input_zonotope ))` is a parameterized function with parameters `self.lambdas`. If it returns 0, then the query is true.
    `analyze()` optimizes these parameters to minimize the loss.

    Attributes:
        znet (zNet): the network with zonotope variables and parameters lambda, s.t self.__net is self.znet "in the concrete"
        input_zonotope (Zonotope): the zonotope to analyze (derived from inp and eps in the __init__)
        true_label (int): the true label of the input point

    TODO: these two attributes will be removed once we use pytorch optimizer instead of home-made gradient descent
        learning_rate (float): the learning rate for gradient descent in `analyze()`
        delta (float): the tolerance threshold for gradient descent in `analyze()`

        __net (networks.FullyConnected || networks.Conv): the network to be analyzed (first layer: Normalization). kept for convenience
        __inp (torch.Tensor): a copy of the input point inp. kept for convenience
        __eps (float): a copy of the queried eps. kept for convenience

    Args:
        net: see Attributes
        inp (torch.Tensor): input point around which to analyze, of shape torch.Size([1, 1, 28, 28])
        eps (float): epsilon, > 0, eps.shape = inp.shape
        true_label (int): see Attributes

    TODO: these two attributes will be removed once we use pytorch optimizer instead of home-made gradient descent
        learning_rate (float, optional): see Attributes
        delta (float, optional): see Attributes
    """

    def __init__(self, net, inp, eps, true_label, learning_rate=1e-2, delta=1e-9):
        self.__net = net
        for p in net.parameters():
            p.requires_grad = False  # avoid doing useless computations
        self.__inp = inp
        self.__eps = eps
        self.true_label = true_label
        self.learning_rate = learning_rate
        self.delta = delta

        self.znet = zNet(net)

        upper = inp + eps
        lower = inp - eps
        upper.clamp_(max=1)  # clip input_zonotope to the input space
        lower.clamp_(min=0)
        a0 = (upper + lower) / 2  # center of the zonotope
        # A must have shape (nb_error_terms, *[shape of input])
        # for the input layer, there is 1 error term for each pixel, so nb_error_terms = inp.numel()

        # A = torch.zeros(784, 1, 28, 28)
        # mask = torch.ones(1, 28, 28, dtype=torch.bool)
        A = torch.zeros(inp.numel(), *inp.shape[1:])
        mask = torch.ones(*inp.shape[1:], dtype=torch.bool)

        A[:, mask] = torch.diag(((upper - lower) / 2).reshape(-1))
        self.input_zonotope = Zonotope(a0=a0, A=A)

    def loss(self, output_zonotope): # TODO: (globally speaking,) using type annotations would probably reduce source code verbosity...
        """Elements x in the last (concrete) layer correspond to logits.
        Args:
            output_zonotope (Zonotope)

        Returns the sum of violations (cf formulas.pdf):
            max_{x(=logit) in output_zonotope} sum_{label l s.t logit[l] > logit[true_label]} (logit[l] - logit[true_label])
        """
        # TODO: we can use https://github.com/szagoruyko/pytorchviz to visualize loss() too.
        assert output_zonotope.dim == torch.Size(10)
        return (output_zonotope - output_zonotope[self.true_label]).relu().sum().upper()

    def analyze(self, verbose=False):
        """Run an optimizer on `self.znet.lambdas` to minimize `self.loss(self.foward())`."""
        # TODO: (top priority) this is where I'm at right now: check out https://pytorch.org/docs/stable/optim.html and set the thing up

        # TODO: check that this is what we want (i.e the set of all the lambdas)
        print(self.znet.parameters() )

        # TODO: choose optimizer and parameters (this one was chosen at random in the doc) https://pytorch.org/docs/stable/optim.html
        optimizer = optim.SGD(self.znet.parameters(), lr=0.01, momentum=0.9)

        optimizer.zero_grad()
        output_zonotope = self.znet(self.input_zonotope, verbose=verbose)
        loss = self.loss(output_zonotope)
        loss.backward()
        optimizer.step()


        loss = self.loss(self.forward())


        while True:
            loss = self.loss(self.forward())
            # TODO floating point problems?
            if loss == 0:
                return True
            self.net.zero_grad()
            loss.backward()
            max_change = 0
            for lambda_layer in self.lambdas:
                grad = lambda_layer.grad
                with torch.no_grad():
                    lambda_layer -= grad * self.learning_rate
                    lambda_layer = torch.max(torch.zeros_like(lambda_layer),
                                             torch.min(torch.ones_like(lambda_layer), lambda_layer))
                lambda_layer.requires_grad_()
                max_change = max(max_change, torch.max(grad))
                # lambda_layer.grad.zero_()
            if max_change < self.delta:
                break
