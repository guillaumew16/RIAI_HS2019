import torch
import torch.nn as nn
import torch.optim as optim

from zonotope import Zonotope
from networks import Normalization

from znet import zNet, zLoss, zMaxSumOfViolations

class Analyzer:
    """
    Analyzer expected by `verifier.py`, to be run using `Analyzer.analyze()`.
    In terms of the attributes, the query to be answered is:
        ?"forall x in input_zonotope, net(x) labels x as true_label"?

    `loss(forward( input_zonotope ))` is a parameterized function with parameters `self.lambdas`. If it returns 0, then the query is true.
    `analyze()` optimizes these parameters to minimize the loss.

    Attributes:
        znet (zNet): the network with zonotope variables and parameters lambda, s.t self.__net is self.znet "in the concrete"
        zloss (zLoss): the loss function, with zonotope input, that translates the logical property to analyze
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
    """

    def __init__(self, net, inp, eps, true_label):
        self.__net = net
        for p in net.parameters():
            p.requires_grad = False  # freeze the concrete network layers, to avoid doing useless computations
        self.__inp = inp
        self.__eps = eps
        self.true_label = true_label

        self.znet = zNet(net)
        self.zloss = zMaxSumOfViolations(nb_classes=10)

        upper = inp + eps
        lower = inp - eps
        upper.clamp_(max=1)  # clip input_zonotope to the input space
        lower.clamp_(min=0)
        a0 = (upper + lower) / 2  # center of the zonotope
        # A must have shape (nb_error_terms, *[shape of input])
        # for the input layer, there is 1 error term for each pixel, so nb_error_terms = inp.numel()

        # A = torch.zeros(784, 1, 28, 28)
        # mask = torch.ones(1, 28, 28, dtype=torch.bool)
        A = torch.zeros(inp.numel(), *inp.shape[1:])
        mask = torch.ones(*inp.shape[1:], dtype=torch.bool)

        A[:, mask] = torch.diag(((upper - lower) / 2).reshape(-1))
        self.input_zonotope = Zonotope(A, a0)

    def analyze(self, verbose=False):
        """Run an optimizer on `self.znet.lambdas` to minimize `self.zloss(self.znet(self.input_zonotope))`.
        Returns True iff the `self.__net` is verifiably robust on `self.input_zonotope`, i.e there exist lambdas s.t loss == 0
        Doesn't return until it is the case, i.e never returns False
        TODO: The last half of the above statement is not exactly it: we can still use ensembling ideas as described in project statement"""
        if verbose: print("entering Analyzer.analyze() with znet: \n{}".format(self.znet))

        # TODO: move this to a unittest or something. Anyway, this has been tested and it works. (DEBUG)
        # a check that self.znet.lambdas is what we want (i.e the set of all the lambdas used as parameters)
        # this also checks that self.znet only has the lambdas as active parameters (i.e all others have required_grad=False)
        # for zlayer in self.znet.zlayers:
        #     print(zlayer)
        #     for p in zlayer.parameters():
        #         # with torch.no_grad(): # to see that self.znet.lambdas does indeed reference the same thing
        #         #     p.fill_(1)
        #         if p.requires_grad == True:
        #             print(p)
        #         else:
        #             print("some frozen parameter (with requires_grad = False) of shape {}".format(p.shape))
        # for lam in self.znet.lambdas:
        #     # with torch.no_grad(): # to see that self.znet.layers.parameters() does indeed reference the same thing
        #     #     p.fill_(2)
        #     print(lam)

        # FOR DEBUG:
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
        # import sys
        # sys.exit()

        # TODO: select optimizer and parameters https://pytorch.org/docs/stable/optim.html. E.g: 
        # optimizer = optim.SGD(self.znet.parameters(), lr=0.01, momentum=0.9)
        print([self.zloss.logit_lambdas, *self.znet.lambdas])
        optimizer = optim.Adam([self.zloss.logit_lambdas, *self.znet.lambdas], lr=0.1)
        # optimizer = optim.Adam(self.znet.lambdas, lr=0.01)

        dataset = [self.input_zonotope] # TODO: can run the optimizer on different zonotopes in general
                                        # e.g we could try partitioning the zonotopes into smaller zonotopes and verify them separately
        for inp_zono in dataset:
            if verbose: print("Analyzer.analyze(): performing the optimization on inp_zono: {}".format(inp_zono))
            # aaaand actually for now just run this optimizer ad infinitum.
            # TODO: do something smarter
            while_counter = 0
            while True:
                print("optimizer parameters", optimizer.__getstate__()['param_groups'][0]['params'])  # DEBUG
                if verbose:
                    print("Analyzer.analyze(): iteration #{}".format(while_counter))
                while_counter += 1

                optimizer.zero_grad()
                out_zono = self.znet(inp_zono, verbose=verbose)
                loss = self.zloss(out_zono)
                if loss == 0: # TODO: floating point problems?
                    return True
                if verbose:
                    print("Analyzer.analyze(): current loss:", loss.item())
                    print("Analyzer.analyze(): doing loss.backward() and optimizer.step()")
                loss.backward()
                optimizer.step()

                # for DEBUG: end the analysis early
                # let run for break_at_iter steps
                break_at_iter = 20
                if while_counter < break_at_iter:
                    continue
                print()
                print("out_zono.A:\n{}\nout_zono.a0:\n{}".format(out_zono.A, out_zono.a0)) # since we're exiting, we can afford to print the results without cluttering the stdout
                # cannot use optimizer.state_dict bc it hides information (returns index of param instead of param tensor). had to hack into pytorch.optimize source code to find this
                print("optimizer parameters", optimizer.__getstate__()['param_groups'][0]['params']) 
                print("self.zloss.logit_lambdas", self.zloss.logit_lambdas)
                print("self.znet.lambdas", self.znet.lambdas)
                print("For convenience in testing, we exit now, even though we still have time before timeout.")
                return False


    # DEBUG: obviously, this shouldn't be run in prod
    def run_in_parallel(self, inp_zono=None, concrete_net=None):
        """A debugging utility. Runs a concrete network and the corresponding zNetwork in parallel to make some checks."""
        if inp_zono is None:
            inp_zono = self.input_zonotope
        if concrete_net is None:
            net = self.__net
            znet = self.znet
        else:
            net = concrete_net
            znet = zNet(concrete_net)
        # print("net(inp_zono.a0) (ground truth):\n", net(inp_zono.a0))
        # print("znet(inp_zono).a0:\n", znet(inp_zono).a0)
        # assert torch.allclose(out_zono.a0, self.__net(inp_zono.a0))
        next_point = inp_zono.a0
        next_zono = inp_zono
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
            # print("next_point - next_zono.a0:\n", next_point - next_zono.a0)


    def make_dot_loss(self, gv_filename):
        """Use https://github.com/szagoruyko/pytorchviz to visualize the computation graph of the loss.
        Writes the result to gv_filename in .gv and .gv.pdf formats.
        Returns the corresponding graphviz.Digraph object."""
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
