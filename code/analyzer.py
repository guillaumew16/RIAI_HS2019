import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization


class Analyzer:
    """
    Analyzer expected by `verifier.py`, to be run using `Analyzer.analyze()`.
    In terms of the attributes, the query to be answered is:
        ?"forall x in input_zonotope, net(x) labels x as true_label"?

    `loss(forward( input_zonotope ))` is a parameterized function with parameters `self.lambdas`. If it returns 0, then the query is true.
    `analyze()` optimizes these parameters to minimize the loss.

    Attributes:
        net (networks.FullyConnected || networks.Conv): the network to be analyzed (first layer: Normalization)
        true_label (int): the true label of the input point
        learning_rate (float, optional): TODO: figure out what this is
        delta (float, optional): TODO: figure out what this is
        input_zonotope (Zonotope): the zonotope to analyze (derived from inp and eps in the __init__)
        lambdas (list of float): the list of the analyzer's parameters lambdas (one `lambda_layer` tensor for each ReLU layer)

        __relu_counter (int): during .forward(), counts ReLU layers to keep track of where we are in the net. kept for convenience
        __inp (torch.Tensor): a copy of the input point inp. kept for convenience
        __eps (float): a copy of the queried eps. kept for convenience

    Args:
        net: see Attributes
        inp (torch.Tensor): input point around which to analyze, of shape torch.Size([1, 1, 28, 28])
        eps (float): epsilon, > 0, eps.shape = inp.shape
        true_label: see Attributes
        learning_rate: see Attributes
        delta: see Attributes
    """
    def __init__(self, net, inp, eps, true_label, learning_rate=1e-1, delta=1e-9):
        self.net = net
        for p in net.parameters():
            p.requires_grad = False # avoid doing useless computations
        self.__inp = inp
        self.__eps = eps
        self.true_label = true_label
        self.learning_rate = learning_rate
        self.delta = delta
        self.__relu_counter = 0

        upper = inp + eps
        lower = inp - eps
        upper.clamp_(max=1) # clip input_zonotope to the input space
        lower.clamp_(min=0)
        a0 = (upper + lower) / 2 # center of the zonotope
        # A must have shape (nb_error_terms, *[shape of input])
        # for the input layer, there is 1 error term for each pixel, so nb_error_terms = inp.numel()
        A = torch.zeros(784, 1, 28, 28)
        mask = torch.ones(1, 28, 28, dtype=torch.bool)
        A[:, mask] = torch.diag( ((upper - lower) / 2).reshape(-1) )
        self.input_zonotope = Zonotope(a0=a0, A=A)

        self.lambdas = []
        # the lambdas are initialized to what the vanilla DeepPoly transformations do
        # so we apply the vanilla DeepPoly to a copy of self.input_zonotope
        init_zonotope = self.input_zonotope.reset()
        for layer in self.net.layers:
            if isinstance(layer, nn.ReLU):
                lam = init_zonotope.compute_lambda_breaking_point()
                self.lambdas.append(lam)
            init_zonotope = self.forward_step(init_zonotope, layer)
        for lambda_layer in self.lambdas:
            lambda_layer.requires_grad_()

    def loss(self, output):
        """The zonotope Z=`output` is at the last layer, so elements x in Z correspond to logits.
        Returns the sum of violations (cf formulas.pdf):
            max_{x(=logit) in output_zonotope} sum_{label l s.t logit[l] > logit[true_label]} (logit[l] - logit[true_label])
        """
        return (output - output[self.true_label]).relu().sum().upper()

    def forward_step(self, inp_zono, layer):
        """Applies `layer` to `inp_zono` (using `self.lambdas` if `layer` is a ReLU) and returns the result.
        Args:
            inp_zono (Zonotope): the input zonotope
            layer (nn.ReLU || nn.Linear || nn.Conv2d || nn.Flatten || Normalization): the layer to apply
        """
        if isinstance(layer, nn.ReLU):
            out = inp_zono.relu(self.lambdas[self.__relu_counter])
            self.__relu_counter += 1
        elif isinstance(layer, nn.Linear):
            out = inp_zono.linear_transformation(layer.weight, layer.bias)
        elif isinstance(layer, nn.Conv2d):
            out = inp_zono.convolution(layer)
        elif isinstance(layer, Normalization):
            out = inp_zono.normalization(layer)
        elif isinstance(layer, nn.Flatten):
            out = inp_zono.flatten()
        return out

    def forward(self):
        """Resets self.input_zonotope and run the analyzer from scratch, using parameters `self.lambdas`."""
        self.__relu_counter = 0
        out = self.input_zonotope.reset()
        for layer in self.net.layers:
            out = self.forward_step(out, layer)
        return out

    def analyze(self):
        """Run gradient descent on `self.lambdas` to minimize `self.loss(self.foward())`."""
        result = False
        while not result:
            loss = self.loss(self.forward())

            # TODO floating point problems?
            if loss == 0:
                result = True
                break
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
        return result
