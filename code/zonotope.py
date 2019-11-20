import torch


class Zonotope:
    def __init__(self, A, a0):
        self.a0 = a0
        self.A = A

    def __add__(self, other):
        if isinstance(other, Zonotope):
            return Zonotope(self.A + other.A, self.a0 + other.a0)
        else:
            return Zonotope(self.A, self.a0 + other)

    def __neg__(self):
        return Zonotope(-self.A, -self.a0)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        return Zonotope(self.A * other, self.a0 * other)

    def __getitem__(self, item):
        return Zonotope(self.A[:, item], self.a0[:, item])

    def sum(self):
        return Zonotope(self.A.sum(1, keepdim=True), self.a0.sum(1, keepdim=True))

    def flatten(self):
        return Zonotope(torch.nn.Flatten()(self.A), torch.nn.Flatten()(self.a0))

    def matmul(self, other):
        return Zonotope(self.A.matmul(other), self.a0.matmul(other))

    def normalization(self, normalization_layer):
        return (self - normalization_layer.mean) * (1 / normalization_layer.sigma)

    def convolution(self, convolution):
        return Zonotope(convolution(self.A), convolution(self.a0))

    def lower(self):
        return self.a0 + (self.A * (-torch.sign(self.A))).sum(0)

    def upper(self):
        return self.a0 + (self.A * torch.sign(self.A)).sum(0)

    def linear_transformation(self, W, b):
        return self.matmul(W.t()) + b

    def relu(self, lambdas):
        l = self.lower()
        u = self.upper()

        lower_map = (u <= 0)[0]
        upper_map = (l >= 0)[0]
        intersection_map = ((l < 0) * (u > 0))[0]
        new_epslon_size = torch.nonzero(intersection_map).size(0)
        new_A_shape = (self.A.shape[0] + new_epslon_size, *self.A.shape[1:])
        A = torch.zeros(new_A_shape)
        a0 = self.a0.clone()
        mu = torch.zeros(a0.shape)

        a0[:, lower_map] = 0
        A[:, lower_map] = 0

        A[:self.A.shape[0], upper_map] = self.A[:, upper_map]

        breaking_point = u[:, intersection_map] / (u[:, intersection_map] - l[:, intersection_map])
        use_l_map = lambdas[:, intersection_map] >= breaking_point

        tmp = torch.zeros(mu[:, intersection_map].shape)
        tmp[use_l_map] = - l[:, intersection_map][use_l_map] * lambdas[:, intersection_map][use_l_map]
        tmp[~ use_l_map] = u[:, intersection_map][~ use_l_map] * (1 - lambdas[:, intersection_map][~ use_l_map])

        mu[:, intersection_map] = tmp / 2

        a0[:, intersection_map] = a0[:, intersection_map] * lambdas[:, intersection_map] + mu[:, intersection_map]
        A[:self.A.shape[0], intersection_map] = self.A[:, intersection_map] * lambdas[:, intersection_map]
        A[self.A.shape[0]:, intersection_map] = torch.diag((mu[:, intersection_map]).reshape(-1))
        return Zonotope(A, a0)
