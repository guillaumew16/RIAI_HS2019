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
        self.__relu_counter = 0
        self.init()

    def init(self):
        init_zonotope = self.input_zonotope.reset()
        for layer in self.net.layers:
            if isinstance(layer, nn.ReLU):
                lam = init_zonotope.compute_lambda_breaking_point()
                lam.requires_grad_()
                self.lambdas.append(lam)
            init_zonotope = self.forward_step(init_zonotope, layer)

    def clip_input_zonotope(self):
        upper = torch.min(self.inp + self.epsilon, torch.ones(self.inp.shape))
        lower = torch.max(self.inp - self.epsilon, torch.zeros(self.inp.shape))
        self.inp = (upper + lower) / 2
        A_shape = (self.inp.numel(), *self.inp[0].shape)
        A = torch.zeros(A_shape)
        A[:, :] = torch.diag(((upper - lower) / 2).reshape(-1))

        return Zonotope(a0=self.inp, A=A)

    def loss(self, output):
        return (output - output[self.true_label]).relu().sum().upper()

    def forward_step(self, inp, layer):
        if isinstance(layer, nn.ReLU):
            out = inp.relu(self.lambdas[self.__relu_counter])
            self.__relu_counter += 1
        elif isinstance(layer, nn.Linear):
            out = inp.linear_transformation(layer.weight, layer.bias)
        elif isinstance(layer, nn.Conv):
            out = inp.convolution(layer)
        elif isinstance(layer, Normalization):
            out = inp.normalization(layer)
        elif isinstance(layer, nn.Flatten):
            out = inp.flatten()
        else:
            raise Exception("Layer not implemented")
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
            for i in range(len(self.lambdas)):
                grad = self.lambdas[i].grad
                self.lambdas[i] = self.lambdas[i] - grad * self.learning_rate
                max_change = max(max_change, torch.max(grad))
            if max_change < self.delta:
                break
        return result
