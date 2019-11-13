import torch


class Zonotope:
    def __init__(self, A, a0):
        self.a0 = a0
        self.A = A

    def __add__(self, other):
        return Zonotope(self.A + other.A, self.a0 + other.a0)

    def __sub__(self, other):
        return Zonotope(self.A - other.A, self.a0 - other.a0)

    def __mul__(self, other):
        return Zonotope(self.A * other.A, self.a0 * other.a0)

    def matmul(self, other):
        # TODO check order
        return Zonotope(torch.matmul(other, self.A), torch.matmul(other, self.a0))

    def convolution(self, convolution):
        return Zonotope(convolution(self.A), convolution(self.a0))

    def lower(self):
        return self.a0 + (self.A * torch.sign(self.A)).sum(0)

    def upper(self):
        return self.a0 + (self.A * (-torch.sign(self.A))).sum(0)

    def relu(self, lambdas):
        l = self.lower()
        u = self.upper()
        intersect = (l >= 0 >= u)
        new_epslon_size = len(intersect)
        new_A_shape = (self.A.shape[0] + new_epslon_size, *self.A.shape[1:])
        A = torch.zeros(new_A_shape)
        a0 = self.a0.clone()
        mu = torch.zeros(a0.shape)

        a0[:, u <= 0] = 0
        A[:, u <= 0] = 0

        A[:self.A.shape[0], l >= 0] = self.A[:, l >= 0]
        mu[:, intersect] = u[:, intersect] * (1 - lambdas[intersect])
        a0[:, intersect] = a0[:, intersect] * lambdas[intersect] + mu[:, intersect]
        A[:self.A.shape[0], intersect] = self.A[:, intersect] * lambdas[intersect]
        # TODO shapes
        A[self.A.shape[0]:, intersect] = mu[:, intersect]
