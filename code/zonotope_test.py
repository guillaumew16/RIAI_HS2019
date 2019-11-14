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

    def test_multiD(self):
        z = Zonotope(a0=torch.tensor([[0, 0, 0]], dtype=torch.float32), A=torch.tensor([[1, 0.5, 0], [1, 0.5, 1]], dtype=torch.float32))

        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[0, 0, 0]]))))
        self.assertTrue(torch.all(torch.eq(z.A, torch.tensor([[1, 0.5, 0], [1, 0.5, 1]]))))
        self.assertTrue(torch.all(torch.eq(z.lower(), torch.tensor([[-2, -1, -1]]))))
        self.assertTrue(torch.all(torch.eq(z.upper(), torch.tensor([[2, 1, 1]]))))

        W = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        z = z.matmul(W)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[0, 0, 0]]))))
        self.assertTrue(torch.all(torch.eq(z.A, torch.tensor([[1, 0.5, 0], [1, 0.5, 1]]))))
        self.assertTrue(torch.all(torch.eq(z.lower(), torch.tensor([[-2, -1, -1]]))))
        self.assertTrue(torch.all(torch.eq(z.upper(), torch.tensor([[2, 1, 1]]))))

        W = torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, -1]], dtype=torch.float32)
        b = torch.tensor([1, -1, 0], dtype=torch.float32)
        z = z.linear_transformation(W, b)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[1, -1, 0]]))))
        self.assertTrue(torch.all(torch.eq(z.A, torch.tensor([[1.5, 0, 0], [2.5, 0, -1]]))))
        self.assertTrue(torch.all(torch.eq(z.lower(), torch.tensor([[-3, -1, -1]]))))
        self.assertTrue(torch.all(torch.eq(z.upper(), torch.tensor([[5, -1, 1]]))))

        z = Zonotope(a0=torch.tensor([[0, 0, 0]], dtype=torch.float32), A=torch.tensor([[1, 0.5, 0], [1, 0.5, 1]], dtype=torch.float32))
        W = torch.tensor([[1, 1, 1], [0, 0, -1]], dtype=torch.float32)
        b = torch.tensor([1, 0], dtype=torch.float32)
        z = z.linear_transformation(W, b)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[1, 0]]))))
        self.assertTrue(torch.all(torch.eq(z.A, torch.tensor([[1.5, 0], [2.5, -1]]))))
        self.assertTrue(torch.all(torch.eq(z.lower(), torch.tensor([[-3, -1]]))))
        self.assertTrue(torch.all(torch.eq(z.upper(), torch.tensor([[5, 1]]))))


if __name__ == '__main__':
    unittest.main()
