import torch
from torch.nn.functional import conv2d


class Zonotope:
    """
    A representation of a zonotope by its center and its error terms coefficients (cf formulas.pdf).
    
    Trivially implements methods `__add__`, `__neg__`, `__sub__`, `__mul__`, `__getitem__`, `sum`,  making e.g Analyzer.loss() concise.
    (These are only convenience stuff, not an attempt of "subclassing Tensors". pyTorch functions are always called on the underlying Tensors `a0` and `A`.)
    
    Attributes:
        A (torch.Tensor): the tensor described in formulas.pdf, with shape [nb_error_terms, *<shape of nn layer>]
        a0 (torch.Tensor): the center of the zonotope, with shape [1, *<shape of nn layer>]

    TODO: check the above statement about the shape of A and a0
    TODO: it seems wasteful that a0 is of shape [1, *]. In fact a lot of the nn layer operations can be applied to the joint tensor [A, a0]
    (except for the fact the biases should not be added to A).
    """
    def __init__(self, A, a0):
        self.a0 = a0
        self.A = A

    def __str__(self):
        return "Zonotope center: {}, epsilon coefficients: {}".format(self.a0, self.A)

    def reset(self):
        """Returns a fresh Zonotope with the same data but no bindings to any output tensor"""
        return Zonotope(self.A.clone().detach(), self.a0.clone().detach()) # cf doc of torch.Tensor.new_tensor()

    # Convenience methods
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
        return self.a0 - self.A.abs().sum(dim=0)
        # return self.a0 + (self.A * (-torch.sign(self.A))).sum(dim=0)

    def upper(self):
        return self.a0 + self.A.abs().sum(dim=0)
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
        return Zonotope(
            conv2d(self.A, weight=convolution.weight, bias=None, stride=convolution.stride, padding=convolution.padding,
                   dilation=convolution.dilation, groups=convolution.groups),
            conv2d(self.a0, weight=convolution.weight, bias=convolution.bias, stride=convolution.stride,
                   padding=convolution.padding, dilation=convolution.dilation, groups=convolution.groups))

    def matmul(self, other):
        return Zonotope(self.A.matmul(other), self.a0.matmul(other))

    def linear_transformation(self, W, b):
        return self.matmul(W.t()) + b

    def compute_lambda_breaking_point(self):
        """Returns the lambda coefficients used by the vanilla DeepPoly.
        The returned value (`lambda_layer`) is a Tensor of shape [1, <*shape of nn layer>] (same as a0)"""
        l = self.lower()
        u = self.upper()

        # ignore variables don't require a lambda for ReLU transformation
        intersection_map = ((l < 0) * (u > 0))[0] # entries s.t l < 0 < u. (implies u-l > 0 so division safe)

        lambda_layer = torch.zeros(self.a0.shape)
        lambda_layer[:, intersection_map] = u[:, intersection_map] / (u[:, intersection_map] - l[:, intersection_map]) # equivalently, replace "[:,*]" by "[0,*]"
        return lambda_layer

    def relu(self, lambdas=None):
        """Apply a convolution layer to this zonotope.
        Args:
            lambdas (torch.Tensor || None): the lambdas to use, of shape [1, <*shape of nn layer>]. 
                If None, do the vanilla DeepPoly transformation.
        """
        l = self.lower()
        u = self.upper()

        lower_map = (u <= 0)[0]
        upper_map = (l >= 0)[0]
        intersection_map = ((l < 0) * (u > 0))[0]

        new_eps_size = torch.nonzero(intersection_map).size(0)
        new_A_shape = (self.A.shape[0] + new_eps_size, *self.A.shape[1:])
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
