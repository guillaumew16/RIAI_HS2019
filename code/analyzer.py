import torch
import torch.nn as nn
import torch.optim as optim

from zonotope import Zonotope
from networks import Normalization

from znet import zNet, zMaxSumOfViolations, zSumOfMaxIndividualViolations, zMaxViolation


class Analyzer:
    """
    Analyzer expected by `verifier.py`, to be run using `Analyzer.analyze()`.
    In terms of the attributes, the query to be answered is:
        ?"forall x in input_zonotope, net(x) labels x as true_label"?
    `self.zloss(self.znet(self.input_zonotope))` is a parameterized loss function with parameters `self.znet.lambdas` and `self.zloss.logit_lambdas`. 
        If it returns a value <= 0, then the query is true.
    `analyze()` optimizes these parameters to minimize the loss.

    Attributes:
        znet (zNet): the network with zonotope variables and parameters lambda, s.t self.__net is self.znet "in the concrete"
        zloss (znet.zLoss): the loss function, with zonotope input, that translates the logical property to analyze
        input_zonotope (Zonotope): the zonotope to analyze (derived from inp and eps in the __init__)
        true_label (int): the true label of the input point
        __net (networks.FullyConnected || networks.Conv): the network to be analyzed (first layer: Normalization). kept for convenience
        __inp (torch.Tensor): a copy of the input point inp. kept for convenience
        __eps (float): a copy of the queried eps. kept for convenience

    Args:
        net: see Attributes
        inp (torch.Tensor): input point around which to analyze, of shape torch.Size([1, 1, 28, 28])
        eps (float): epsilon, > 0, eps.shape = inp.shape
        true_label (int): see Attributes
        nb_classes (int, optional): the number of classes to analyze, default=10. Allows to do testing without breaking the code.
    """

    def __init__(self, net, inp, eps, true_label, nb_classes=10):
        self.__net = net
        for p in net.parameters():
            p.requires_grad = False  # freeze the concrete network layers, to avoid doing useless computations
        self.__inp = inp
        self.__eps = eps
        self.true_label = true_label

        self.znet = zNet(net, input_shape=inp.shape[1:], nb_classes=nb_classes)
        # self.zloss = zMaxSumOfViolations(true_label=true_label, nb_classes=nb_classes)
        self.zloss = zSumOfMaxIndividualViolations(true_label=true_label, nb_classes=nb_classes)
        # self.zloss = zMaxViolation(true_label=true_label, nb_classes=nb_classes)

        upper = inp + eps
        lower = inp - eps
        upper.clamp_(max=1)  # clip input_zonotope to the input space
        lower.clamp_(min=0)
        a0 = (upper + lower) / 2  # center of the zonotope
        # A must have shape (nb_error_terms, *[shape of input])
        # for the input layer, there is 1 error term for each pixel, so nb_error_terms = inp.numel()
        A = torch.zeros(inp.numel(), *inp.shape[1:])  # torch.zeros(784, 1, 28, 28)
        mask = torch.ones(*inp.shape[1:], dtype=torch.bool)  # torch.ones(1, 28, 28, dtype=torch.bool)
        A[:, mask] = torch.diag(((upper - lower) / 2).reshape(-1))
        self.input_zonotope = Zonotope(A, a0)

    def analyze(self, verbose=False):
        """Returns True iff the `self.__net` is verifiably robust on `self.input_zonotope`
        Doesn't return until it is proved, i.e never returns False

        What the current implementation does:
            Run an optimizer on `self.znet.lambdas` and `self.zloss.logit_lambdas` to minimize `self.zloss(self.znet(self.input_zonotope))`.
            Return True when we find values of the lambdas s.t loss == 0
        """
        if verbose: print("entering Analyzer.analyze() with znet: \n{}".format(self.znet))

        # useful stuff for debugging
        # self.check_parameters()
        # self.run_in_parallel()
        # class NetByLayers(nn.Module):
        #     def __init__(self,layers):
        #         super().__init__()
        #         self.layers = nn.Sequential(*layers)
        #     def forward(self, x):
        #         return self.layers(x)
        # net_layers = [layer for layer in self.__net.layers]
        # net_layers.append(nn.ReLU())
        # net_layers.append(nn.ReLU())
        # net_layers.append(nn.ReLU())
        # net = NetByLayers(net_layers)
        # self.run_in_parallel(self.input_zonotope, net)

        # TODO: select optimizer and parameters https://pytorch.org/docs/stable/optim.html. E.g:
        # optimizer = optim.SGD(<parameters>, lr=0.01, momentum=0.9)
        if self.zloss.has_lambdas:
            optimizer = optim.Adam([self.zloss.logit_lambdas, *self.znet.lambdas], lr=0.1)
            # zm.zReLU has the feature that setting the requires_gradient to False makes us use DeepZ, e.g:
            # self.zloss.logit_lambdas.requires_grad = False
        else:
            optimizer = optim.Adam([*self.znet.lambdas], lr=0.1)
        
        # WIP (ugly to commit this I know but whatevs)
        # lr_arr = [0.1, 0.1, 0.1, 0.1]
        # assert len(lr_arr) == len(self.znet.lambdas)
        # optimizer = optim.Adam([
        #     { 'params': self.znet.lambdas[i], 'lr': lr_arr[i] }
        #     for i in #range(len(lr_arr))
        # ], lr=0.1)

        dataset = [self.input_zonotope]  # TODO: can run the optimizer on different zonotopes in general
        # e.g we could try partitioning the zonotopes into smaller zonotopes and verify them separately
        for inp_zono in dataset:
            if verbose: 
                print("Analyzer.analyze(): performing the optimization on inp_zono: {}".format(inp_zono))
                print("Analyzer.analyze(): using loss function {} and optimizer {}".format(type(self.zloss), optimizer))
            # aaaand actually for now just run this optimizer ad infinitum. TODO: do something smarter
            while_counter = 0
            while True:
                # print("optimizer parameters", optimizer.__getstate__()['param_groups'][0]['params'])  # useful for debugging
                if verbose:
                    print("Analyzer.analyze(): iteration #{}".format(while_counter))
                optimizer.zero_grad()
                out_zono = self.znet(inp_zono, verbose=verbose)

                loss = self.zloss(out_zono, verbose=verbose)

                if loss <= 0:  # TODO: floating point problems? (there is indeed still a pb here, since zMaxSumOfViolations is non-negative.)
                    if verbose: print("Analyzer.analyze(): found loss<=0 (loss={}) after {} iterations. The property is proved.".format(loss.item(), while_counter))
                    return True
                if verbose:
                    print("Analyzer.analyze(): current loss:", loss.item())
                    print("Analyzer.analyze(): doing loss.backward() and optimizer.step()")
                loss.backward()
                optimizer.step()

                # print() # DEBUG
                # print("after optimizer.step and BEFORE clamping:")
                # for lambda_layer in self.znet.lambdas:
                #     print("lambda layer with shape {}: #(non-zero coefficients) / #(all coefficients) = {}/{}".format(lambda_layer.shape, lambda_layer.nonzero().size(0), lambda_layer.numel() ))
                # # Rk: these numbers don't change much! Why?

                with torch.no_grad(): # we can safely ignore grads here. They will be recomputed from scratch at the next evaluation of the loss anyway.
                    for lambda_layer in self.znet.lambdas:
                        lambda_layer.clamp_(min=0, max=1)
                    if self.zloss.has_lambdas:
                        self.zloss.logit_lambdas.clamp_(min=0, max=1)

                # print("after optimizer.step and AFTER clamping:") # DEBUG
                # for lambda_layer in self.znet.lambdas:
                #     print("lambda layer with shape {}: #(non-zero coefficients) / #(all coefficients) = {}/{}".format(lambda_layer.shape, lambda_layer.nonzero().size(0), lambda_layer.numel() ))
                # print()

                while_counter += 1

                """
                # for DEBUG: end the analysis early
                # let run for break_at_iter steps
                break_at_iter = 20
                if while_counter < break_at_iter:
                    continue
                print()
                print("out_zono.A:\n{}\nout_zono.a0:\n{}".format(out_zono.A, out_zono.a0)) # since we're exiting, we can afford to print the results without cluttering the stdout
                # cannot use optimizer.state_dict bc it hides information (returns index of param instead of param tensor). Had to look into pytorch.optimize source code to find this.
                print("optimizer parameters:\n", optimizer.__getstate__()['param_groups'][0]['params'])
                # print("self.zloss.logit_lambdas:\n{}\nself.znet.lambdas:\n{}".format(self.zloss.logit_lambdas, self.znet.lambdas)) # should contain the same thing as what we just printed
                print("For convenience in testing, we exit now (after {} iterations), even though we still have time before timeout.".format(break_at_iter))
                return False
                """

        import warnings
        warnings.warn("Analyzer.analyze() is returning False, i.e. we're giving up even though we haven't timed out.")
        return False

    # Debugging utilities (DEBUG: obviously, none of these should be run in prod)
    # ~~~~~~~~~~~~~~~~~~~

    def check_parameters(self):
        """A debugging utility.
        Check that the self.znet.lambdas is what we want (i.e the set of all the lambdas used as parameters).
        This also checks that self.znet only has the lambdas as active parameters (i.e all others have required_grad=False).
        """
        import zmodules as zm
        for zlayer in self.znet.zlayers:
            print(zlayer)
            for p in zlayer.parameters():
                # with torch.no_grad(): # to see that self.znet.lambdas does indeed reference the same thing (the next for loop will print all 1's)
                #     p.fill_(1)
                if p.requires_grad == True:
                    assert isinstance(zlayer, zm.zReLU)
                    print(p)
                else:
                    print("some frozen parameter (with requires_grad = False) of shape {}".format(p.shape))
        for lam in self.znet.lambdas:
            print(lam)

    def run_in_parallel(self, inp_zono=None, concrete_net=None):
        """A debugging utility. 
        Runs a concrete network and the corresponding zNetwork in parallel to make some checks.
        Note that the concrete point and the zonotope center are NOT theoretically supposed to be identical, since we recenter the zonotope at each ReLU.
        """
        if inp_zono is None:
            inp_zono = self.input_zonotope
        if concrete_net is None:
            net = self.__net
            znet = self.znet
        else:
            net = concrete_net
            znet = zNet(concrete_net)
        next_point = inp_zono.a0
        next_zono = inp_zono
        assert torch.allclose(next_point, next_zono.a0)  # at the input layer the concrete point and the zono-center are identical
        for i in range(len(net.layers)):
            layer = net.layers[i]
            zlayer = znet.zlayers[i]
            print(layer, zlayer)
            next_point = layer(next_point)
            next_zono = zlayer(next_zono)
            # print(next_point.shape, next_zono.a0.shape)
            print("next_point (ground truth):\n", next_point)
            print("next_zono.a0:\n", next_zono.a0)
            print("next_zono.A:\n", next_zono.A)

    def make_dot_loss(self, gv_filename):
        """Use https://github.com/szagoruyko/pytorchviz to visualize the computation graph of the loss.
        Writes the result to gv_filename in .gv and .gv.pdf formats.
        Returns the corresponding graphviz.Digraph object.
        """
        try:
            import torchviz
            inp_zono = Zonotope(torch.zeros_like(self.input_zonotope.Z))
            out_zono = self.znet(inp_zono)
            loss = self.zloss(out_zono)
            dot = torchviz.make_dot(loss)
            dot.render(gv_filename, view=False)
            return dot
        except ImportError as err:
            import warnings
            warnings.warn("torchviz is not installed in the execution environment, so cannot make_dot. Skipping")

    def make_dot_znet(self, gv_filename):
        """Visualize the computation graph of the zNet."""
        try:
            import torchviz
            inp_zono = Zonotope(torch.zeros_like(self.input_zonotope.Z))
            out_zono = self.znet(inp_zono)
            out_aggr = torch.cat([out_zono.A, out_zono.a0], dim=0)
            dot = torchviz.make_dot(out_aggr)
            dot.render(gv_filename, view=False)
            return dot
        except ImportError as err:
            import warnings
            warnings.warn("torchviz is not installed in the execution environment, so cannot make_dot. Skipping")

    def make_dot_concrete(self, gv_filename):
        """Visualize the computation graph of concrete network `self.__net`."""
        try:
            import torchviz
            inp = torch.zeros(784, 1, 28, 28)
            for p in self.__net.parameters():
                p.requires_grad = True  # temporarily set it back to true (required for the computation graph)
            out = self.__net(inp)
            for p in self.__net.parameters():
                p.requires_grad = False
            dot = torchviz.make_dot(out)
            dot.render(gv_filename, view=False)
            return dot
        except ImportError as err:
            import warnings
            warnings.warn("torchviz is not installed in the execution environment, so cannot make_dot. Skipping")
