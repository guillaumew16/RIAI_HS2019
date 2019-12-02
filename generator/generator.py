import argparse
import os
import random

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../code') # little hack. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
from networks import FullyConnected, Conv

from adv import pgd_untargeted

DEVICE = 'cpu'
INPUT_SIZE = 28

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
args = parser.parse_args()

BASE_DIR_PATH = '../test_cases_generated/' + args.net
NUM_EXAMPLES_TO_GENERATE = args.num
DO_SANITY_CHECK = args.sc

if args.net == 'fc1':
    net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
elif args.net == 'fc2':
    net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
elif args.net == 'fc3':
    net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
elif args.net == 'fc4':
    net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
elif args.net == 'fc5':
    net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
elif args.net == 'conv1':
    net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
elif args.net == 'conv2':
    net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
elif args.net == 'conv3':
    net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
elif args.net == 'conv4':
    net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
elif args.net == 'conv5':
    net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))


DATASET_PATH = '../mnist_data'

mnist_dataset = datasets.MNIST(DATASET_PATH, train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))
mnist_loader = torch.utils.data.DataLoader(mnist_dataset, shuffle=True)



def generate_uid():
    uids = []
    with os.scandir(BASE_DIR_PATH+"/maybe_robust") as it:
        for f_name in it:
            under_score_pos = f_name.name.rfind('_')
            if under_score_pos == -1:
                raise Warning(BASE_DIR_PATH+"/maybe_robust contains a file with bad filename: "+f_name+" (filename should contain '_')")
                continue
            uids.append( f_name.name[0:under_score_pos] )
    with os.scandir(BASE_DIR_PATH+"/not_robust") as it:
        for f_name in it:
            under_score_pos = f_name.name.rfind('_')
            if under_score_pos == -1:
                raise Warning(BASE_DIR_PATH+"/not_robust contains a file with bad filename: "+f_name+" (filename should contain '_')")
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
        raise ValueError("bad tensor shape: expected x of shape [1, 1, 28, 28], got "+str(x.shape))
    filename = str(uid) + "_" + str(eps) + ".txt"
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

# for testing purposes (check that write_to_file then read_from_file returns the original tensor)
def read_from_file(filename):
    """Takes a file and returns the datapoint represented.
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
        raise ValueError("bad file path (should contain 'maybe_robust' or 'not_robust'): " + str(filename))

    # keep the exact same code as in `verifier.py`
    with open(filename, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(filename[:-4].split('/')[-1].split('_')[-1])
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)

    print("Finished reading.")
    return inputs, eps, true_label, robust

# for testing purposes
def display_image(x, title=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(x[0, 0, :, :], vmin=0, vmax=1, cmap='gray')
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    plt.show()



gen = generate_uid()
# parameters for PGD:
eps_step_ratio = 0.01   # eps_step / eps, where eps_step is the step size in FGSM (see adv.py)
k = 1000                # nb of projections in PGD
num_restarts = 100      # number of times to try PGD with a different random seed

for idx, (x, true_label) in enumerate(mnist_loader): # batch_size=1 by default
    if idx >= NUM_EXAMPLES_TO_GENERATE:
        break
    eps = random.uniform(0.005, 0.2) # eps ranges between 0.005 and 0.2
    eps_step = eps * eps_step_ratio

    # make sure that the network correctly labels x, else drop this image (don't produce it as test case)
    # indeed, if we kept such test cases, they would immediately trigger an assert in verifier.py anyway
    # (this means that less than NUM_EXAMPLES_TO_GENERATE examples will actually be generated, but whatever)
    outs = net(x)
    pred_label = outs.max(dim=1)[1].item()
    if pred_label != true_label:
        print("The image x is incorrectly labeled by net. Dropping this case and moving on...")
        continue

    # look for an adversarial example
    robust = True
    for _ in range(num_restarts): # try multiple times (adv.pgd_ includes some randomness)
        x_adv = pgd_untargeted(net, x, true_label, k, eps, eps_step, device='cpu', clip_min=0, clip_max=1) # candidate adversarial example
        outs = net(x_adv)
        pred_label = outs.max(dim=1)[1].item()
        if pred_label != true_label:
            robust = False
            break
    filename = write_to_file(x, eps, true_label.item(), robust, next(gen))

    if DO_SANITY_CHECK:
        display_image(x, title="original image (x)")
        x_read, _, _, _ = read_from_file(filename)
        print(x.shape)
        print(x_read.shape)
        display_image(x_read, title="image written (x_read)")
        print("filename:", filename)
        print("max difference between `x` and `x_read`:", (x-x_read).max() )
        print()

