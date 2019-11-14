import torch


class Zonotope:
    def __init__(self, A, a0):
        self.a0 = a0
        self.A = A

    def __add__(self, other):
        if isinstance(other, Zonotope):
            return Zonotope(self.A + other.A, self.a0 + other.a0)
        else:
            return Zonotope(self.A + other, self.a0 + other)

    def __neg__(self):
        return Zonotope(-self.A, -self.a0)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, Zonotope):
            return Zonotope(self.A * other.A, self.a0 * other.a0)
        else:
            return Zonotope(self.A * other, self.a0 * other)

    def matmul(self, other):
        return Zonotope(torch.matmul(other, self.A), torch.matmul(other, self.a0))

    def convolution(self, convolution):
        return Zonotope(convolution(self.A), convolution(self.a0))

    def lower(self):
        return self.a0 + (self.A * (-torch.sign(self.A))).sum(0)

    def upper(self):
        return self.a0 + (self.A * torch.sign(self.A)).sum(0)

    def linear(self, W, b):
        return self.matmul(torch.transpose(W, 0, 1)) + b

    def relu(self, lambdas):
        l = self.lower()
        u = self.upper()
        intersect = ((l <= 0) * (u >= 0))[0]
        new_epslon_size = torch.nonzero(intersect).size(0)
        new_A_shape = (self.A.shape[0] + new_epslon_size, *self.A.shape[1:])
        A = torch.zeros(new_A_shape)
        a0 = self.a0.clone()
        mu = torch.zeros(a0.shape)

        a0[:, (u <= 0)[0]] = 0
        A[:, (u <= 0)[0]] = 0

        A[:self.A.shape[0], (l >= 0)[0]] = self.A[:, (l >= 0)[0]]
        mu[:, intersect] = u[:, intersect] * (1 - lambdas[:, intersect])
        a0[:, intersect] = a0[:, intersect] * lambdas[:, intersect] + mu[:, intersect]
        A[:self.A.shape[0], intersect] = self.A[:, intersect] * lambdas[:, intersect]
        A[self.A.shape[0]:, intersect] = torch.diag((mu[:, intersect]).reshape(-1))
        return Zonotope(A, a0)
