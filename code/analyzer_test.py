import unittest
import torch
import torch.nn as nn
from analyzer import Analyzer

class MockNet(nn.Module):
    def __init__(self, input_size):
        super(MockNet, self).__init__()

        layers = [nn.Linear(input_size, 10), nn.ReLU()]
        self.layers = nn.Sequential(*layers)


class AnalyzerTest(unittest.TestCase):
    def test_input_clipping(self):
        x = torch.tensor([[1, 0.98, 0.05, 0.5]])
        epsilon = 0.06
        a = Analyzer(MockNet(x.shape[1]), x, epsilon, None)
        self.assertTrue(torch.all(torch.isclose(a.inp, torch.tensor([[0.97, 0.96, 0.055, 0.5]]))))
        self.assertTrue(torch.all(torch.isclose(a.epsilon, torch.tensor([[0.03, 0.04, 0.055, 0.06]]))))

if __name__ == '__main__':
    unittest.main()
