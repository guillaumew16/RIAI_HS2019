import unittest
from zonotope import Zonotope
from networks import Normalization
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
        z = Zonotope(a0=torch.tensor([[0, 0, 0]], dtype=torch.float32),
                     A=torch.tensor([[1, 0.5, 0], [1, 0.5, 1]], dtype=torch.float32))

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

        z = Zonotope(a0=torch.tensor([[0, 0, 0]], dtype=torch.float32),
                     A=torch.tensor([[1, 0.5, 0], [1, 0.5, 1]], dtype=torch.float32))
        W = torch.tensor([[1, 1, 1], [0, 0, -1]], dtype=torch.float32)
        b = torch.tensor([1, 0], dtype=torch.float32)
        z = z.linear_transformation(W, b)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[1, 0]]))))
        self.assertTrue(torch.all(torch.eq(z.A, torch.tensor([[1.5, 0], [2.5, -1]]))))
        self.assertTrue(torch.all(torch.eq(z.lower(), torch.tensor([[-3, -1]]))))
        self.assertTrue(torch.all(torch.eq(z.upper(), torch.tensor([[5, 1]]))))

        z = Zonotope(a0=torch.tensor([[1, 2, 1]], dtype=torch.float32),
                     A=torch.tensor([[1, 0.5, 0], [1, 0.5, 1]], dtype=torch.float32))
        self.assertEqual(z[0].a0, torch.tensor([[1]]))
        self.assertTrue(torch.all(torch.eq(z[0].A, torch.tensor([[1], [1]]))))
        self.assertEqual(z.sum().a0, torch.tensor([[4]]))
        self.assertTrue(torch.all(torch.eq(z.sum().A, torch.tensor([[1.5], [2.5]]))))

    def test_convolution(self):
        a0 = torch.ones(1, 1, 3, 3)
        A = torch.tensor(
            [[[[0., 1., 0.],
               [1., 0.3, 1.],
               [1., 1., 1.]]],
             [[[0., 0., 0.],
               [1., 0.3, 1.],
               [1., 1., 1.]]],
             [[[0., -1., 0.],
               [1., 0.3, 1.],
               [1., 1., 1.]]]])
        z = Zonotope(a0=a0, A=A)
        conv = torch.nn.Conv2d(1, 1, 3)
        torch.nn.init.ones_(conv.weight)
        torch.nn.init.ones_(conv.bias)
        z = z.convolution(conv)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[[[10.]]]]))))
        self.assertTrue(torch.all(torch.eq(z.A, torch.tensor([[[[6.3]]], [[[5.3]]], [[[4.3]]]]))))

    def test_flatten(self):
        a0 = torch.ones(1, 1, 2, 2)
        A = torch.tensor(
            [[[[0., 0.9],
               [1., 0.3]]],
             [[[-2., 0.1],
               [2., 0.3]]]])
        z = Zonotope(a0=a0, A=A)
        z = z.flatten()
        self.assertEqual(z.a0.shape, (1, 4))
        self.assertEqual(z.A.shape, (2, 4))

    def test_normalization(self):
        z = Zonotope(a0=torch.tensor([[0.1307, 0.1307, 0.1307]], dtype=torch.float32),
                     A=torch.tensor([[0.3081, 0.3081, 0.3081], [0.3081, 0.3081, 0.3081]], dtype=torch.float32))
        z = z.normalization(Normalization(device="cpu"))
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[0, 0, 0]]))))
        self.assertTrue(torch.all(torch.eq(z.A, torch.tensor([[1, 1, 1], [1, 1, 1]]))))

    def test_relu(self):
        z = Zonotope(a0=torch.tensor([[0, 1, -2]], dtype=torch.float32),
                     A=torch.tensor([[1, 0.5, 0.5], [1, 0.5, 1]], dtype=torch.float32))
        lambdas = torch.zeros(z.a0.shape)
        z = z.relu(lambdas)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[1, 1, 0]], dtype=torch.float32))))
        self.assertTrue(
            torch.all(torch.eq(z.A, torch.tensor([[0, 0.5, 0], [0, 0.5, 0], [1.0, 0, 0]], dtype=torch.float32))))

        z = Zonotope(a0=torch.tensor([[0, 1, -2]], dtype=torch.float32),
                     A=torch.tensor([[1, 0.5, 0.5], [1, 0.5, 1]], dtype=torch.float32))
        lambdas = torch.ones(z.a0.shape)
        z = z.relu(lambdas)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[1, 1, 0]], dtype=torch.float32))))
        self.assertTrue(
            torch.all(torch.eq(z.A, torch.tensor([[1, 0.5, 0], [1, 0.5, 0], [1.0, 0, 0]], dtype=torch.float32))))

        z = Zonotope(a0=torch.tensor([[0, 1, -2]], dtype=torch.float32),
                     A=torch.tensor([[1, 0.5, 0.5], [1, 0.5, 1]], dtype=torch.float32))
        lambdas = torch.ones(z.a0.shape) / 2
        z = z.relu(lambdas)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[0.5, 1, 0]], dtype=torch.float32))))
        self.assertTrue(
            torch.all(torch.eq(z.A, torch.tensor([[0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0, 0]], dtype=torch.float32))))

        a0 = torch.ones(1, 1, 2, 2)
        A = torch.tensor(
            [[[[0., 0.9],
               [1., 0.3]]],
             [[[-2., 0.1],
               [2., 0.3]]]])
        z = Zonotope(a0=a0, A=A)
        lambdas = torch.ones(a0.shape) / 2
        z = z.relu(lambdas)
        self.assertTrue(torch.all(torch.eq(z.a0, torch.tensor([[[[1.25, 1], [1.5, 1]]]], dtype=torch.float32))))
        self.assertTrue(torch.all(torch.eq(z.A, torch.tensor(
            [[[[0., 0.9],
               [0.5, 0.3]]],
             [[[-1, 0.1],
               [1, 0.3]]],
             [[[0.75, 0],
               [0, 0]]],
             [[[0, 0],
               [1, 0]]]], dtype=torch.float32))))


if __name__ == '__main__':
    unittest.main()
