import torch
from torch.nn.functional import conv2d


class Zonotope:
    """
    A representation of a zonotope by its center and its error terms coefficients (cf formulas.pdf).

    Trivially implements methods `__add__`, `__neg__`, `__sub__`, `__mul__`, `__getitem__`, `sum`,  making e.g Analyzer.loss() concise.
    (These are only convenience stuff, not an attempt of "subclassing Tensors". pyTorch functions are always called on the underlying Tensors `a0` and `A`.)

    Attributes:
        Z (torch.Tensor): tensor with shape [1 + nb_error_terms, *<shape of nn layer>].
            Z[0] corresponds to the center of the zonotope (a0 in the old version).
            Z[1:] corresponds to the epsilon-coefficient tensor (A in the old version).
    
    Old attributes, now accessible as getters:
        A (torch.Tensor): the tensor described in formulas.pdf, with shape [nb_error_terms, *<shape of nn layer>]
            Note that we do NOT support constant zonotopes, i.e there must be >= 1 error terms. (Almost everything would work, but relu() would be even trickier.)
        a0 (torch.Tensor): the center of the zonotope, with shape [1, *<shape of nn layer>]

    Args:
        A_or_Z (torch.Tensor): if the constructor is called with only one argument, then A_or_Z represent Z. Otherwise it represents A.
        a0 (torch.Tensor, optional)
    """

    def __init__(self, A_or_Z, a0=None):
        if a0 is None:
            self.Z = A_or_Z
        else:
            assert a0.shape[1:] == A_or_Z.shape[1:]
            self.Z = torch.cat([a0, A_or_Z], dim=0)
        assert self.nb_error_terms >= 1

    @property
    def a0(self):
        return self.Z[:1]
    @property
    def A(self):
        return self.Z[1:]

    def __str__(self):
        return "Zonotope with dim={} and nb_error_terms={}".format(self.dim, self.nb_error_terms)
        # return "Zonotope center: {}, epsilon coefficients: {}".format(self.a0, self.A)

    @property
    def dim(self):
        """Dimension of the ambient space, i.e shape of nn layer"""
        return self.Z.shape[1:]
    @property
    def nb_error_terms(self):
        return self.Z.shape[0] - 1

    def reset(self):
        """Returns a fresh Zonotope with the same data but no bindings to any output tensor"""
        return Zonotope(self.Z.clone().detach())  # cf doc of torch.Tensor.new_tensor()

    # Convenience methods
    # ~~~~~~~~~~~~~~~~~~~

    def __add__(self, other):
        if isinstance(other, Zonotope):
            assert self.nb_error_terms == other.nb_error_terms
            assert other.dim == torch.Size([1]) or self.dim == other.dim  # keep tight control over the broadcasting magic, bc I'm not certain how it works
            return Zonotope(self.Z + other.Z)
        else:
            assert not isinstance(other, torch.Tensor) or other.numel() == 1 \
                or other.shape == self.dim or other.shape == self.a0.shape  # keep tight control over the broadcasting magic
            return Zonotope(self.A, self.a0 + other)

    def __neg__(self):
        return Zonotope(-self.Z)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        """Multiplication by a constant"""
        return Zonotope(self.Z * other)

    def __getitem__(self, item):
        """For a flat zonotope (i.e living in a 'flattened' space), returns the zonotope of the `item`-th variable."""
        if len(self.dim) != 1:
            raise UserWarning("Called Zonotope.__getitem__ on an instance with dim={}. \
                It should only be called on instances living in 'flattened spaces', i.e with dim of the form torch.Size([n]).".format(self.dim))
        return Zonotope(self.Z[:, item].reshape(-1, 1))

    def sum(self):
        """Returns the zonotope of the sum of all variables."""
        dims_to_reduce = list( range(1, len(self.Z.shape)) )  # sum over all dimensions except the first one (corresponding to error terms' index)
        return Zonotope(self.Z.sum(dims_to_reduce, keepdim=True))

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
        return Zonotope(self.Z.flatten(start_dim=1))  # preserve error-term-index dimension. (strictly equivalent to torch.nn.Flatten()(self.Z))

    def normalization(self, mean, sigma):
        """Apply a normalization layer to this zonotope.
        Args:
            mean (torch.Tensor): mean to subtract, of shape [1, 1, 1, 1] (same as in networks.Normalization). (Any shape broadcastable to [1] works too.)
            sigma (torch.Tensor): sigma to divide, of shape [1, 1, 1, 1]
        """
        return Zonotope(self.A / sigma, (self.a0 - mean) / sigma)
        # return (self - mean) * (1 / sigma)

    def convolution(self, conv):
        """Apply a convolution layer to this zonotope.
        Args:
            conv (torch.nn.Conv2d)"""
        # TODO: do some tests to see if one is faster than the other. (my guess is that it changes almost nothing.)
        # Implementation 1:
        res_Z = conv2d(self.Z, weight=conv.weight, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        to_add = torch.zeros_like(res_Z)
        to_add[0] = conv.bias.view(*conv.bias.shape, 1, 1) # encapsulates in single pixels (height and width 1)
        return Zonotope(res_Z + to_add)
        # Implementation 2:
        # return Zonotope(
        #     conv2d(self.A, weight=conv.weight, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups),
        #     conv2d(self.a0, weight=conv.weight, bias=conv.bias, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        # )

    def linear_transformation(self, linear):
        """Apply a linear layer to this zonotope.
        Args: 
            linear (nn.Linear): the linear layer with the same weight and bias as the corresponding concrete layer.
                In fact, using the concrete layer itself is just fine."""
        # TODO: do some tests to see if one is faster than the other. (my guess is that it changes almost nothing.)
        # Implementation 1:
        res_Z = self.Z.matmul(linear.weight.t())
        to_add = torch.zeros_like(res_Z)
        to_add[0] = linear.bias
        return Zonotope(res_Z + to_add)
        # Implementation 2:
        # return Zonotope(
        #     self.A.matmul(linear.weight.t()),
        #     linear(self.a0)
        # )

    def compute_lambda_breaking_point(self):
        """Returns the lambda coefficients used by the vanilla DeepZ.
        The returned value (`lambda_layer`) is a Tensor of shape [1, <*shape of nn layer>] (same as a0)"""
        l = self.lower()
        u = self.upper()

        # construct update_map, a boolean mask of shape [1, <*shape of nn layer>], i.e the same as lambda_layer, u, l
        # Option 1: intersection map, i.e we ignore variables don't require a lambda for ReLU transformation (set the corresponding lambda to 0)
        intersection_map = ((l < 0) * (u > 0))  # entries s.t l < 0 < u. (implies u-l > 0 so division safe)
        # Option 2: full initialization map, i.e just make sure the division is safe
        nonzero_map = (u - l != 0)

        # TODO: test whether using the full initialization map helps. So far I found one (and only one) test case where it did (fc5 on fc5/img0)
        # update_map = intersection_map
        update_map = nonzero_map

        lambda_layer = torch.zeros([1, *self.dim])
        lambda_layer[update_map] = u[update_map] / (
                    u[update_map] - l[update_map])  # equivalently, replace "[:,*]" by "[0,*]"
        return lambda_layer

    # TODO: see if we can use the Z form instead of distinguishing A and a0, as we did for the other transformations.
    # Not much hope though, and arguably it's normal: relu is the hard case.
    def relu(self, lambdas=None):
        """Apply a ReLU layer to this zonotope.
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

        # If lambda is None we use the vanilla DeepZ transformation, which means lambda = breaking_point.
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
