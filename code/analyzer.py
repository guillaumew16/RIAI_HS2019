import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization


class Analyzer:
    """
    Analyzer expected by verifier.py, to be run using Analyzer.analyze()

    Attributes:
        net (networks.Conv || networks.FullyConnected): the network to be analyzed
        inp (torch.Tensor): input point (x0 in formulas.pdf), inp.shape matches net's input layer
        eps (float): epsilon, > 0
        true_label (int): the true label of inp
        learning_rate (float, optional): TODO: figure out what this is
        delta (float, optional): TODO: figure out what this is

        input_zonotope (Zonotope): TODO: figure out what this is
        lambdas (list of float): TODO: figure out what this is
        __relu_counter (int): TODO: figure out what this is

    """
    def __init__(self, net, inp, eps, true_label, learning_rate=1e-1, delta=1e-9):
        self.net = net
        for p in net.parameters():
            p.requires_grad = False
        self.inp = inp
        self.eps = eps
        self.true_label = true_label
        self.learning_rate = learning_rate
        self.delta = delta

        # clip input_zonotope
        upper = torch.min(self.inp + self.eps, torch.ones(self.inp.shape))
        lower = torch.max(self.inp - self.eps, torch.zeros(self.inp.shape))
        self.inp = (upper + lower) / 2
        A = torch.zeros(self.inp.numel(), *self.inp[0].shape)
        A[:, self.inp[0] == self.inp[0]] = torch.diag(((upper - lower) / 2).reshape(-1))
        self.input_zonotope = Zonotope(a0=self.inp, A=A)

        self.lambdas = []
        self.__relu_counter = 0

        init_zonotope = self.input_zonotope.reset()
        for layer in self.net.layers:
            if isinstance(layer, nn.ReLU):
                lam = init_zonotope.compute_lambda_breaking_point()
                self.lambdas.append(lam)
            init_zonotope = self.forward_step(init_zonotope, layer)
        for lambda_layer in self.lambdas:
            lambda_layer.requires_grad_()

    def loss(self, output):
        return (output - output[self.true_label]).relu().sum().upper()

    def forward_step(self, inp, layer):
        if isinstance(layer, nn.ReLU):
            out = inp.relu(self.lambdas[self.__relu_counter])
            self.__relu_counter += 1
        elif isinstance(layer, nn.Linear):
            out = inp.linear_transformation(layer.weight, layer.bias)
        elif isinstance(layer, nn.Conv2d):
            out = inp.convolution(layer)
        elif isinstance(layer, Normalization):
            out = inp.normalization(layer)
        elif isinstance(layer, nn.Flatten):
            out = inp.flatten()
        return out

    def forward(self):
        self.__relu_counter = 0
        out = self.input_zonotope.reset()
        for layer in self.net.layers:
            out = self.forward_step(out, layer)
        return out

    def analyze(self):
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
