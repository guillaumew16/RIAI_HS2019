import torch
import torch.nn as nn

from zonotope import Zonotope
from networks import Normalization


class Analyzer:
    def __init__(self, net, inp, eps, true_label, learning_rate=1e-3, delta=1e-9):
        self.net = net
        self.inp = inp
        self.epsilon = eps
        self.input_zonotope = self.clip_input_zonotope()
        self.true_label = true_label
        self.learning_rate = learning_rate
        self.delta = delta
        self.lambdas = []
        for layer in net.layers:
            if isinstance(layer, nn.ReLU):
                lam = torch.zeros(layer.size())
                lam.requires_grad_()
                self.lambdas.append(lam)

    def clip_input_zonotope(self):
        upper = torch.min(self.inp + self.epsilon, torch.ones(self.inp.shape))
        lower = torch.max(self.inp - self.epsilon, torch.zeros(self.inp.shape))
        self.inp = (upper + lower) / 2
        self.epsilon = (upper - lower) / 2
        return Zonotope(self.inp, torch.ones(self.inp.shape) * self.epsilon)

    def loss(self, output):
        return (output - output[self.true_label]).relu().sum().upper()

    def forward(self):
        out = self.input_zonotope.reset()
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
            for lamda_layer in self.lambdas:
                lamda_layer = lamda_layer - self.learning_rate * lamda_layer.grad
                max_change = max(max_change, self.learning_rate * lamda_layer.grad)
            if max_change < self.delta:
                break
        return result
