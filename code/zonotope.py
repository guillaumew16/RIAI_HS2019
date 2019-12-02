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

    TODO: it seems wasteful that a0 is of shape [1, *]. In fact a lot of the nn layer operations can be applied to the joint tensor [A, a0]
    (except for the fact the biases should not be added to A).
    """

    def __init__(self, A, a0):
        self.a0 = a0
        self.A = A

    def __str__(self):
        return "Zonotope center: {}, epsilon coefficients: {}".format(self.a0, self.A)

    @property
    def dim(self):
        """Dimension of the ambiant space"""
        return A.shape[1:]

    @property
    def nb_error_terms(self):
        return A.shape[0]

    def reset(self):
        """Returns a fresh Zonotope with the same data but no bindings to any output tensor"""
        return Zonotope(self.A.clone().detach(), self.a0.clone().detach())  # cf doc of torch.Tensor.new_tensor()

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
        """For a flat zonotope (i.e living in a 'flattened' space with shape (n,) ), returns the zonotope of the `item`-th variable."""
        if len(self.a0.shape) > 2:
            raise Warning("Called Zonotope.__getitem__ on an instance with a0.shape=" + str(a0.shape)
                + ". It should only be called on instances living in 'flattened spaces', i.e with a0.shape of the form [1, n].")
        return Zonotope(self.A[:, item:item + 1], self.a0[:, item:item + 1])

    def sum(self):
        """For a flat zonotope (i.e living in a 'flattened' space with shape (n,) ), returns the zonotope of the sum of all variables."""
        if len(self.a0.shape) > 2:
            raise Warning("Called Zonotope.__getitem__ on an instance with a0.shape=" + str(a0.shape)
                + ". It should only be called on instances living in 'flattened spaces', i.e with a0.shape of the form [1, n].")
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
        """Returns the lambda coefficients used by the vanilla DeepZ.
        The returned value (`lambda_layer`) is a Tensor of shape [1, <*shape of nn layer>] (same as a0)"""
        l = self.lower()
        u = self.upper()

        # ignore variables don't require a lambda for ReLU transformation
        intersection_map = ((l < 0) * (u > 0))[0]  # entries s.t l < 0 < u. (implies u-l > 0 so division safe)

        lambda_layer = torch.zeros(self.a0.shape)
        lambda_layer[:, intersection_map] = u[:, intersection_map] / (
                    u[:, intersection_map] - l[:, intersection_map])  # equivalently, replace "[:,*]" by "[0,*]"
        return lambda_layer

    def relu(self, lambdas=None):
        """Apply a convolution layer to this zonotope.
        Args:
            lambdas (torch.Tensor || None): the lambdas to use, of shape [1, <*shape of nn layer>].
                If None, do the vanilla DeepZ transformation.
        """
        # compute lower and upper bound of the layer
        l = self.lower()
        u = self.upper()

        # compute boolean map to find which elements of the input have a negative lower bound and a positive upper bound
        lower_map = (u <= 0)[0]
        upper_map = (l >= 0)[0]
        intersection_map = ((l < 0) * (u > 0))[0]

        # We need to add a new epslon for each element that must be approximated by the relu transformation.
        # new_eps_size is this number so the new shape of A is self.A.shape[0] + new_eps_size in the first dimension
        # and the same as prevoius A in the other dimensions.
        new_eps_size = torch.nonzero(intersection_map).size(0)
        new_A_shape = (self.A.shape[0] + new_eps_size, *self.A.shape[1:])
        A = torch.zeros(new_A_shape)
        a0 = self.a0.clone()
        mu = torch.zeros(a0.shape)

        # When the upper bound is lower than zero, we know that the output is always 0,
        # so both the center and the coefficients are zero.
        # The coefficients are already initialized as 0 so we don't need to do it again.
        a0[:, lower_map] = 0

        # When the input is always positive, the output of this relu is the same as the input,
        # the center is already initialized this way so we only need to set the coefficients.
        A[:self.A.shape[0], upper_map] = self.A[:, upper_map]

        # From now on whe compute the relu approximation for the point that have a negative lower bound
        # and a positive upper bound. So we will use the intersection map to access the tensors only on these points.

        # Compute the coefficient of the lines that intersect both (l, 0) and (u, u) for each element.
        # When lambda is greater than this value the only line that we can use is the one passing through (l, 0)
        # otherwise we can only use the one passing through (u, u).
        breaking_point = u[:, intersection_map] / (u[:, intersection_map] - l[:, intersection_map])

        # y = lambda*x + mu + mu*eps_new where mu = d / 2 and d is the translation of the line that approximate
        # the upper bound of the relu transformation.
        # So a0 = a0*lambda + mu and A = lambda*A for all the previous epslons and A = mu for the new epslons.

        # If lambda is None we use the vanilla DeepPoly transformation, which means lambda = breaking_point.
        if lambdas is None:
            mu[:, intersection_map] = - l[:, intersection_map] * breaking_point / 2

            # Compute new a0 and A from x.
            a0[:, intersection_map] = a0[:, intersection_map] * breaking_point + mu[:, intersection_map]
            A[:self.A.shape[0], intersection_map] = self.A[:, intersection_map] * breaking_point
        else:
            # Boolean map that decide when to use the (l, 0) point.
            use_l_map = lambdas[:, intersection_map] >= breaking_point

            # Compute d as described before and mu = d / 2
            tmp = torch.zeros(mu[:, intersection_map].shape)
            tmp[use_l_map] = - l[:, intersection_map][use_l_map] * lambdas[:, intersection_map][use_l_map]
            tmp[~ use_l_map] = u[:, intersection_map][~ use_l_map] * (1 - lambdas[:, intersection_map][~ use_l_map])

            mu[:, intersection_map] = tmp / 2

            # Compute new a0 and A from x.
            a0[:, intersection_map] = a0[:, intersection_map] * lambdas[:, intersection_map] + mu[:, intersection_map]
            A[:self.A.shape[0], intersection_map] = self.A[:, intersection_map] * lambdas[:, intersection_map]

        # Finally compute values for the coefficients of the new epsilons.
        # Because each epsilon has one non zero coefficient and 0 for all other elements
        # this line create a diagonal matrix to set the values of A.
        A[self.A.shape[0]:, intersection_map] = torch.diag((mu[:, intersection_map]).reshape(-1))
        return Zonotope(A, a0)
