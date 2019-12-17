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
                    help='Neural network to generate test cases for.')
parser.add_argument('--num',
                    type=int,
                    default=1,
                    help='Number of new test cases to generate. (default: 1)')
parser.add_argument('--sc', 
                    action='store_true',
                    help="Display and sanity-check that each file produced is correct.")
parser.add_argument('--method', 
                    type=str,
                    choices=['my_pgd', 'art_carlini', 'art_pgd'], # TODO: add more methods
                    default='art_carlini',
                    help="Method to use to generate adversarial examples. (default: ART's PGD attack)")
parser.add_argument('--eps',
                    type=float,
                    default=0.15,
                    help='Maximum epsilon, strictly between 0.0 and 0.2. (default: 0.15)')
parser.add_argument('--nro',
                    action='store_true',
                    help='"Not Robust Only". Only save the not_robust test cases.')
args = parser.parse_args()

# define "globals"
NET_NAME = args.net
BASE_DIR_PATH = '../test_cases_generated/' + args.net
NUM_EXAMPLES_TO_GENERATE = args.num
DO_SANITY_CHECK = args.sc
ATTACK_METHOD = args.method
if args.eps >= 0.2 or args.eps <= 0.0:
    raise UserWarning("Bad value for maximum epsilon: expected float strictly between 0.0 and 0.2, got {}".format(args.eps))
MAX_EPSILON = args.eps
NOT_ROBUST_ONLY = args.nro

DEVICE = 'cpu'
INPUT_SIZE = 28
DATASET_PATH = '../mnist_data'
BATCH_SIZE = min(NUM_EXAMPLES_TO_GENERATE, 5)

# create output dirs if they don't exist yet
os.makedirs(BASE_DIR_PATH+"/maybe_robust", exist_ok=True)
os.makedirs(BASE_DIR_PATH+"/not_robust", exist_ok=True)

def load():
    # load network to attack
    net = get_net(NET_NAME)
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % NET_NAME, map_location=torch.device(DEVICE)))
    for p in net.parameters():
        p.requires_grad = False

    # load dataset
    mnist_dataset = datasets.MNIST(DATASET_PATH, train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]
    ))
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, shuffle=True, batch_size=BATCH_SIZE)
    artDataGenerator = PyTorchDataGenerator(mnist_loader, size=None, batch_size=BATCH_SIZE)

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
                                        eps=MAX_EPSILON,
                                        batch_size=BATCH_SIZE)
        elif ATTACK_METHOD == "art_pgd":
            attacker = ProjectedGradientDescent(classifier,
                                        eps=MAX_EPSILON,
                                        eps_step=0.05,
                                        batch_size=BATCH_SIZE)
        else:
            raise ValueError # the argument parser doesn't allow other values for ATTACK_METHOD
    
    return net, artDataGenerator, attacker



def main():
    net, artDataGenerator, attacker = load()
    uid_gen = generate_uid()

    print("Generating (up to) {} test cases \n \
            for network: {} \n \
            Max eps: {} \n \
            Attack method: {} \n \
            Batch size: {} \n \
            Only keeping not_robust test cases: {}"
        .format(NUM_EXAMPLES_TO_GENERATE, NET_NAME, MAX_EPSILON, ATTACK_METHOD, BATCH_SIZE, NOT_ROBUST_ONLY))

    for batch_i in range(NUM_EXAMPLES_TO_GENERATE // BATCH_SIZE): # Rk: we may generate less than NUM_EXAMPLES_TO_GENERATE examples
        print("Running batch #{}/{}...".format(batch_i+1, NUM_EXAMPLES_TO_GENERATE // BATCH_SIZE))
        # run the attack by batch
        (x_np, y_np) = artDataGenerator.get_batch() # x_np, y_np: np.ndarrays
        if type(attacker) in [CarliniLInfMethod, ProjectedGradientDescent]: # TODO: all attackers should eventually have some randomness for eps (but not all support it yet)
            rand_eps = random.uniform(0.005, MAX_EPSILON)
            print("rand_eps =", rand_eps)
            params = {
                'eps': rand_eps,
                'eps_step': rand_eps/10,
                'num_random_init': 5,
            }
            attacker.set_params(**params)
        x_adv_np = attacker.generate(x_np, y_np)
        # convert the whole batch to torch.tensor
        x_torch = torch.from_numpy(x_np)
        y_torch = torch.from_numpy(y_np)
        x_adv_torch = torch.from_numpy(x_adv_np)
        check_and_save_batch(x_torch, y_torch, x_adv_torch, net, uid_gen, rand_eps)
    
def check_and_save_batch(x_torch, y_torch, x_adv_torch, net, uid_gen, rand_eps):
    # check and save the results, image by image
    for idx in range(BATCH_SIZE):
        x = x_torch[idx].view(1, 1, 28 , 28)
        true_label = y_torch[idx].item()
        x_adv = x_adv_torch[idx].view(1, 1, 28, 28)

        # make sure that the network correctly labels x, else drop this image (don't produce it as test case)
        # indeed, if we kept such test cases, they would immediately trigger an assert in verifier.py anyway
        outs = net(x)
        pred_label = outs.max(dim=1)[1].item()
        if pred_label != true_label:
            print("The image x is incorrectly labeled by net. Dropping this case and moving on...")
            continue

        # re-check whether the x_adv is indeed an adversarial example
        if torch.allclose(x, x_adv, atol=1e-9, rtol=0):
            # when x_adv is exactly equal to x, this is how CarliniLInfMethod signals that we failed to find an adversarial example.
            robust = True
        else:
            outs_adv = net(x_adv)
            label_adv = outs_adv.max(dim=1)[1].item()
            robust = (label_adv == true_label)
        if NOT_ROBUST_ONLY and robust == True:
            print("Failed to find an adversarial example. Flag 'nro' ('not-robust only') was set. Dropping this case and moving on...")
            continue

        # determine eps
        eps = (x - x_adv).abs().max()
        if not robust:
            if eps < 0.001:
                print("WARNING: we claim to have found an adversarial example with eps={}, which is suspiciously low".format(eps))
            eps = eps.item()*1.00001
            eps = max(eps, 0.005) # respect the project specifications
            if eps > 0.2:
                print("Found an adversarial example with eps={} > 0.2, which is not suitable for our case. Dropping this case and moving on...".format(eps))
                continue
        else:
            eps = random.uniform(0.005, rand_eps) # randomize eps a bit. (we didn't find an adversarial example with eps < rand_eps)

        # save the original image as "maybe_robust" or "not_robust"
        filename = write_to_file(x, eps, true_label, robust, next(uid_gen))
        if DO_SANITY_CHECK:
            x_read, _, _, _ = read_from_file(filename)
            assert x.shape == torch.Size([1, 1, 28, 28])
            assert x_read.shape == torch.Size([1, 1, 28, 28])
            # print("max difference between `x` and `x_read`:", (x-x_read).max() )
            display_images([x, x_read, x_adv], ["original image (x)", 
                                                "image written (x_read)",
                                                "adversarial example (x_adv)"])
            assert torch.allclose(x, x_read)



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

def generate_uid():
    """Generate UIDs 'img0', 'img1', ... to easily identify filenames for 
    Filenames for test cases must be of the form 'img<x>_<epsilon>.txt'
    """
    uids = []
    for categ in ["maybe_robust", "not_robust", "verifiable"]:
        with os.scandir(os.path.join(BASE_DIR_PATH, categ)) as it:
            for f_name in it:
                under_score_pos = f_name.name.rfind('_')
                if under_score_pos == -1:
                    warnings.warn("{}/{} contains a file with bad filename: {} (filename should contain '_')".format(BASE_DIR_PATH, categ, f_name))
                    continue
                uids.append( f_name.name[0:under_score_pos] )
    idx = 0
    while True:
        uid = "img" + str(idx) # candidate uid
        if not uid in uids:
            yield uid
        idx += 1

def write_to_file(x, eps, true_label, robust, uid):
    """Writes the test case to a file. The auto-generated filename looks like
        BASE_DIR_PATH/<robustness>/img<uid>_<eps>.txt
    e.g:
        test_cases_generated/fc1/maybe_robust/img0_0.128011.txt

    Args:
        x (torch.Tensor): the MNIST image to verify, of shape [1, 1, 28, 28]
        eps (float): the epsilon
        true_label (int): the true label (between 0 and 9)
        robust (bool): True if PGD didn't find an adversarial example around x, False otherwise (but there may still be one)
        uid (int): a uid to prefix the filename with, for convenience

    Returns:
        filename (str): the auto-generated filename used
    """
    if list(x.shape) != [1, 1, 28, 28]:
        raise ValueError("bad tensor shape: expected x of shape [1, 1, 28, 28], got {}".format(x.shape))
    filename = "{}_{}.txt".format(uid, eps)
    robustness_path = 'maybe_robust' if robust else 'not_robust'
    filename = os.path.join(BASE_DIR_PATH, robustness_path, filename)
    print("Writing data to file", filename, "...")
    with open(filename, 'w') as f:
        f.write(str(true_label) + "\n")
        # write x, one scalar by row
        for i in range(28):
            for j in range(28):
                towrite = x[0, 0, i, j].item()
                f.write(str(towrite) + "\n")
    print("Finished writing.")
    return filename

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

def display_images(x_arr, title_arr=None):
    if title_arr is None:
        title_arr = [None] * len(x_arr)
    fig, ax_arr = plt.subplots(1, len(x_arr))
    for i in range(len(x_arr)):
        x = x_arr[i]
        title = title_arr[i]
        ax = ax_arr[i]
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

    