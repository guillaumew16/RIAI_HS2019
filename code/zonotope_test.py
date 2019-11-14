import unittest
from zonotope import Zonotope
import torch


class ZonotopeTest(unittest.TestCase):
    def test_basic_operations(self):
        z = Zonotope(a0=torch.tensor([[0]]), A=torch.tensor([[1]]))

        self.assertEqual(z.a0, torch.tensor([[0]]))
        self.assertEqual(z.A, torch.tensor([[1]]))
        self.assertEqual(z.lower(), torch.tensor([[-1]]))
        self.assertEqual(z.upper(), torch.tensor([[1]]))

        z = z + 1
        self.assertEqual(z.a0, torch.tensor([[1]]))
        self.assertEqual(z.A, torch.tensor([[1]]))
        self.assertEqual(z.lower(), torch.tensor([[0]]))
        self.assertEqual(z.upper(), torch.tensor([[2]]))

        z = z * 2
        self.assertEqual(z.a0, torch.tensor([[2]]))
        self.assertEqual(z.A, torch.tensor([[2]]))
        self.assertEqual(z.lower(), torch.tensor([[0]]))
        self.assertEqual(z.upper(), torch.tensor([[4]]))


if __name__ == '__main__':
    unittest.main()
