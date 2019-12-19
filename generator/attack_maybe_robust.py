import argparse
import os
import random
import warnings

import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../code') # little hack. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
from networks import FullyConnected, Conv

from art.classifiers import BlackBoxClassifier, PyTorchClassifier
from art.data_generators import PyTorchDataGenerator
from art.utils import to_categorical
from art.attacks import CarliniLInfMethod, ProjectedGradientDescent



# parse input
parser = argparse.ArgumentParser()
parser.add_argument('--net',
                    type=str,
                    choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                    required=True,
                    help='Neural network to attack maybe_robust for.')
parser.add_argument('--num',
                    type=int,
                    default=None,
                    help='Number of maybe_robust test cases to attack. (default: all)')
parser.add_argument('--method', 
                    type=str,
                    choices=['my_pgd', 'art_carlini', 'art_pgd'], # TODO: add more methods
                    default='art_carlini',
                    help="Method to use to generate adversarial examples. (default: ART's PGD attack)")
args = parser.parse_args()

# define "globals"
NET_NAME = args.net
BASE_DIR_PATH = '../test_cases_generated/' + args.net
# BASE_DIR_PATH = '../test_cases/' + args.net
NUM_TO_ATTACK = args.num
ATTACK_METHOD = args.method

DEVICE = 'cpu'
INPUT_SIZE = 28

# create output dirs if they don't exist yet
os.makedirs(BASE_DIR_PATH+"/not_robust", exist_ok=True)

def load():
    # load network to attack
    net = get_net(NET_NAME)
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % NET_NAME, map_location=torch.device(DEVICE)))
    for p in net.parameters():
        p.requires_grad = False

    # make attacker
    if ATTACK_METHOD == "my_pgd":
        attacker = MyPgdAttacker(net, eps_step_ratio=0.01, k=1000, num_restarts=100)
    else:
        # create the ART classifier
        def predict_numpy(x):
            return net.forward( torch.from_numpy(x) ).numpy()
        classifier = PyTorchClassifier(model=net, 
                                        loss=torch.nn.CrossEntropyLoss(), # ART's ProjectedGradientDescent uses this
                                        optimizer=None, # not used (fit, fit_generator, __setstate__ are not called)
                                        input_shape=(1, 28, 28), 
                                        nb_classes=10,
                                        clip_values=(0, 1))
        if ATTACK_METHOD == "art_carlini":
            attacker = CarliniLInfMethod(classifier,
                                        max_iter=100, # default=10
                                        # eps=,       # to be set later
            )
        elif ATTACK_METHOD == "art_pgd":
            attacker = ProjectedGradientDescent(classifier,
                                        # eps=,       # to be set later
                                        # eps_step=,  # to be set later
            )
        else:
            raise ValueError # the argument parser doesn't allow other values for ATTACK_METHOD
    
    return net, attacker



def main():
    print("Attacking (up to) {} maybe_robust test cases \n \
            for network: {} \n \
            Attack method: {}"
        .format(NUM_TO_ATTACK, NET_NAME, ATTACK_METHOD))

    net, attacker = load()

    tried = 0
    with os.scandir(BASE_DIR_PATH+"/maybe_robust") as it:
        for f_name in it:
            inputs, eps, true_label, maybe_robustness = read_from_file(f_name.path)
            assert maybe_robustness == True

            # convert to np.ndarrays and feed to ART attacker
            x_np = inputs.numpy()
            y_np = to_categorical([true_label], nb_classes=10)
            if type(attacker) in [CarliniLInfMethod, ProjectedGradientDescent]: # TODO: all attackers should eventually have some randomness for eps (but not all support it yet)
                params = {
                    'eps': eps,
                    'eps_step': eps/10,
                    'num_random_init': 5,
                }
                attacker.set_params(**params)
            x_adv_np = attacker.generate(x_np, y_np)

            # convert back to torch.tensor
            x_torch = torch.from_numpy(x_np)
            y_torch = torch.from_numpy(y_np)
            x_adv_torch = torch.from_numpy(x_adv_np)
            # check whether attack succeeded
            outs_adv = net(x_adv_torch)
            label_adv = outs_adv.max(dim=1)[1].item()
            if not (x_torch - x_adv_torch).abs().max() <= eps:
                raise UserWarning("The attacker returned a x_adv that is more than eps away from the original image.")
            found_advers = (label_adv != true_label) # and (x_torch - x_adv_torch).abs().max() <= eps

            if found_advers:
                # move f_name to BASE_DIR_PATH+"/not_robust"
                print("found adversarial example")
                src_path = f_name.path
                dst_path = os.path.join(BASE_DIR_PATH, "not_robust", f_name.name)
                print("moving {} to {}".format(src_path, dst_path))
                os.replace(src_path, dst_path)
            else:
                # else, leave it where it is
                print("failed to find an adversarial example :/")

            tried += 1
            if tried == NUM_TO_ATTACK:
                break

# utility functions
def get_net(net_name):
    if net_name == 'fc1':
        return FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif net_name == 'fc2':
        return FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif net_name == 'fc3':
        return FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif net_name == 'fc4':
        return FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'fc5':
        return FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif net_name == 'conv1':
        return Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_name == 'conv2':
        return Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_name == 'conv3':
        return Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif net_name == 'conv4':
        return Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif net_name == 'conv5':
        return Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

# more utility functions, for testing purposes
def read_from_file(filename):
    """Takes a file and returns the datapoint represented. 
    (Useful to check that write_to_file then read_from_file returns the original tensor.)
    Returns:
        x (torch.Tensor): tensor of shape [1, 1, 28, 28]
        eps (float)
        true_label (int)
        robust (bool)
    """
    print("Reading data from file", filename, "...")
    robust = None
    if filename.find('maybe_robust'):
        robust = True
    elif filename.find('not_robust'):
        robust = False
    if robust is None:
        raise ValueError("bad file path (should contain 'maybe_robust' or 'not_robust'): {}".format(filename))

    # keep the exact same code as in `verifier.py`
    with open(filename, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(filename[:-4].split('/')[-1].split('_')[-1])
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

    print("Finished reading.")
    return inputs, eps, true_label, robust

def display_one_image(x, title=None):
    fig, ax = plt.subplots()
    ax.imshow(x[0, 0, :, :], vmin=0, vmax=1, cmap='gray')
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    plt.show()



if __name__ == "__main__":
    main()



# ================
# == my attacks ==
# ================
# should have more or less the same interface as art.attacks.Attack (except for initialization)

"""
Args:
    eps_step_ratio (float): eps_step / eps, where eps_step is the step size in FGSM (see adv.py)
    k (int): nb of projections in PGD
    num_restarts (int): number of times to try PGD with a different random seed
"""
class MyPgdAttacker():
    def __init__(self, eps_step_ratio, k, num_restarts):
        raise NotImplementedError

    