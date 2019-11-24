import unittest
import torch
import torch.nn as nn

from zonotope import Zonotope
from analyzer import Analyzer


class MockNet1(nn.Module):
    def __init__(self, input_size):
        super(MockNet1, self).__init__()

        self.layers = nn.Sequential(nn.Linear(input_size, 10), nn.ReLU())


class MockNet2(nn.Module):
    def __init__(self):
        super(MockNet2, self).__init__()

        self.layers = nn.Sequential(nn.ReLU())


class MockNet3(nn.Module):
    def __init__(self, input_size):
        super(MockNet3, self).__init__()

        lin = nn.Linear(input_size, input_size)
        torch.nn.init.eye_(lin.weight)
        torch.nn.init.constant_(lin.bias, -0.1)

        self.layers = nn.Sequential(lin, nn.ReLU())


class MockNet4(nn.Module):
    def __init__(self, input_size):
        super(MockNet4, self).__init__()

        lin1 = nn.Linear(input_size, input_size)
        torch.nn.init.eye_(lin1.weight)
        torch.nn.init.constant_(lin1.bias, -0.1)

        lin2 = nn.Linear(input_size, input_size)
        torch.nn.init.eye_(lin2.weight)
        torch.nn.init.constant_(lin2.bias, -0.1)

        self.layers = nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU())


class AnalyzerTest(unittest.TestCase):
    def test_input_clipping(self):
        x = torch.tensor([[1, 0.98, 0.05, 0.5]])
        epsilon = 0.06
        a = Analyzer(MockNet1(x.shape[1]), x, epsilon, None)
        self.assertTrue(torch.all(torch.isclose(a.inp, torch.tensor([[0.97, 0.96, 0.055, 0.5]]))))
        self.assertTrue(torch.all(torch.isclose(a.input_zonotope.A, torch.tensor([[0.03, 0, 0, 0],
                                                                                  [0, 0.04, 0, 0],
                                                                                  [0, 0, 0.055, 0],
                                                                                  [0, 0, 0, 0.06]]))))

    def test_lambdas_initialization(self):
        x = torch.tensor([[0.1, 0.5, 0.9]])
        epsilon = 0.05
        z = Zonotope(torch.tensor([[0.05, 0.05, 0.05]]), x)
        a = Analyzer(MockNet2(), x, epsilon, None)
        self.assertTrue(torch.allclose(a.lambdas[0], z.upper() / (z.upper() - z.lower())))

    def test_loss(self):
        x = torch.tensor([[0.1, 0.5, 0.9]])
        a = Analyzer(MockNet2(), x, 0.05, 1)
        loss = a.loss(a.forward())
        self.assertTrue(loss > 0)

        x = torch.tensor([[0.1, 0.5, 0.9]])
        a = Analyzer(MockNet3(x.shape[1]), x, 0.05, 0)

        loss = a.loss(a.forward())
        loss.backward()
        self.assertTrue(loss > 0)

        grad = a.lambdas[0].grad
        self.assertTrue(grad[0, 0] > 0)

        x = torch.tensor([[0.1, 0.5, 0.9]])
        a = Analyzer(MockNet4(x.shape[1]), x, 0.05, 0)

        loss = a.loss(a.forward())
        loss.backward()
        self.assertTrue(loss > 0)

        grad = a.lambdas[0].grad
        self.assertTrue(grad is not None)
        grad = a.lambdas[1].grad
        self.assertTrue(grad is not None)

    def test_analyze(self):
        x = torch.tensor([[0.1, 0.5, 0.9]])
        epsilon = 0.05
        a = Analyzer(MockNet2(), x, epsilon, 2)
        self.assertTrue(a.analyze())

        a = Analyzer(MockNet2(), x, 0, 2)
        self.assertTrue(a.analyze())

        a = Analyzer(MockNet2(), x, 0.05, 1)
        self.assertFalse(a.analyze())

        a = Analyzer(MockNet2(), x, 0.5, 2)
        self.assertFalse(a.analyze())


if __name__ == '__main__':
    unittest.main()
