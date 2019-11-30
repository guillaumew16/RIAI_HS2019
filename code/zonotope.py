import torch


class Zonotope:
    """
    A representation of a zonotope by its center and its error terms coefficients (cf formulas.pdf).
    
    Trivially implements methods `__add__`, `__neg__`, `__sub__`, `__mul__`, `__getitem__`, `sum`,  making e.g Analyzer.loss() concise.
    (These are only convenience stuff, not an attempt of "subclassing Tensors". pyTorch functions are always called on the underlying Tensors `a0` and `A`.)
    
    Also implements sum, flatten, 
    
    Attributes:
        A (torch.Tensor): the tensor described in formulas.pdf, with shape [nb_error_terms, *<shape of nn layer>]
        a0 (torch.Tensor): the center of the zonotope, with shape [*<shape of nn layer>]

    TODO: check shape of A and a0; I think a0 is actually [1, *<shape of nn layer>], when "shape of nn layer" means that
    the corresponding torch.nn Layer is actually of signature [N, *<shape of nn layer>] -> [N, *<shape of nn output layer>] (where N is a batch size)
    --> so, to check in the torch.nn docs.
    """
    def __init__(self, A, a0):
        self.a0 = a0
        self.A = A

    def reset(self):
        """Returns a fresh Zonotope with the same data but no bindings to any output tensor"""
        return Zonotope(self.A.clone().detach(), self.a0.clone().detach()) # cf doc of torch.Tensor.new_tensor()

    # Trivial convenience methods
    # ~~~~~~~~~~~~~~~~~~~

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
        """Multiplication by a constant"""
        return Zonotope(self.A * other, self.a0 * other)

    def __getitem__(self, item):
        """For a flat zonotope (i.e living in a 'flattened' space with shape (n,) ), returns the zonotope of the `item`-th variable
        TODO: check whether it does only work for a flat zonotope"""
        return Zonotope(self.A[:, item:item + 1], self.a0[:, item:item + 1])

    def sum(self):
        """For a flat zonotope (i.e living in a 'flattened' space with shape (n,) ), returns the zonotope of the sum of all variables
        TODO: check whether it does only work for a flat zonotope"""
        return Zonotope(self.A.sum(1, keepdim=True), self.a0.sum(1, keepdim=True))

    def lower(self):
        return self.a0 + self.A.abs().sum(dim=0)
        # return self.a0 + (self.A * (-torch.sign(self.A))).sum(dim=0)

    def upper(self):
        return self.a0 - self.A.abs().sum(dim=0)
        # return self.a0 + (self.A * torch.sign(self.A)).sum(dim=0)

    # Handle the application of torch.nn layers
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def flatten(self):
        """Apply a torch.nn.Flatten() layer to this zonotope."""
        return Zonotope(torch.nn.Flatten()(self.A), torch.nn.Flatten()(self.a0))

    def normalization(self, normalization_layer):
        """Apply a normalization layer to this zonotope.
        Args:
            normalization_layer (networks.Normalization)"""
        return (self - normalization_layer.mean) * (1 / normalization_layer.sigma)

    def convolution(self, convolution):
        """Apply a convolution layer to this zonotope.
        Args:
            convolution (torch.nn.Conv2d)"""
        return Zonotope(convolution(self.A), convolution(self.a0))

    def matmul(self, other):
        return Zonotope(self.A.matmul(other), self.a0.matmul(other))

    def linear_transformation(self, W, b):
        return self.matmul(W.t()) + b

    def compute_lambda_breaking_point(self):
        l = self.lower()
        u = self.upper()
        zero_map = (u - l != 0)[0]
        lambdas = torch.zeros(self.a0.shape)
        lambdas[:, zero_map] = u[:, zero_map] / (u[:, zero_map] - l[:, zero_map])
        return lambdas

    def relu(self, lambdas=None):
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
        if lambdas is None:
            mu[:, intersection_map] = l[:, intersection_map] * breaking_point / 2

            a0[:, intersection_map] = a0[:, intersection_map] * breaking_point + mu[:, intersection_map]
            A[:self.A.shape[0], intersection_map] = self.A[:, intersection_map] * breaking_point
        else:
            use_l_map = lambdas[:, intersection_map] >= breaking_point

            tmp = torch.zeros(mu[:, intersection_map].shape)
            tmp[use_l_map] = - l[:, intersection_map][use_l_map] * lambdas[:, intersection_map][use_l_map]
            tmp[~ use_l_map] = u[:, intersection_map][~ use_l_map] * (1 - lambdas[:, intersection_map][~ use_l_map])

            mu[:, intersection_map] = tmp / 2

            a0[:, intersection_map] = a0[:, intersection_map] * lambdas[:, intersection_map] + mu[:, intersection_map]
            A[:self.A.shape[0], intersection_map] = self.A[:, intersection_map] * lambdas[:, intersection_map]

        A[self.A.shape[0]:, intersection_map] = torch.diag((mu[:, intersection_map]).reshape(-1))
        return Zonotope(A, a0)

    def __str__(self):
        return "Zonotope center: {}, epsilon coefficients: {}".format(self.a0, self.A)
