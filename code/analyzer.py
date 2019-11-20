import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization


class Analyzer:
    def __init__(self, net, inp, eps, true_label):
        self.net = net
        self.inp = inp
        self.eps = eps
        self.input_zonotope = Zonotope(self.inp, torch.ones(self.inp.shape) * self.eps)
        self.true_label = true_label
        self.lambdas = []
        for layer in net.layers:
            if isinstance(layer, nn.ReLU):
                self.lambdas.append(torch.zeros(layer.size()))


    def loss(self, output):
        """
        ot = output[self.true_label]  # TODO not implemented in Zonotope
        return (ot - output).relu().sum().upper()  # TODO sum over the axes
        """




    def forward(self):
        out = self.input_zonotope
        for layer in self.net.layers:
            if isinstance(layer, nn.ReLU):
                out = out.relu()
            elif isinstance(layer, nn.Linear):
                out = out.linear_transformation(layer.weight, layer.bias)
            elif isinstance(layer, nn.Conv):
                out = out.convolution(layer)
            elif isinstance(layer, Normalization):
                out = out.normalization(layer)
            elif isinstance(layer, nn.Flatten):
                out = out.flatten()

    def analyze(self):
        return False