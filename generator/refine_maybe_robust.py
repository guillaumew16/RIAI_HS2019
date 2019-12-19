import argparse
import os
import random
import warnings
import signal

import torch
# from torchvision import datasets, transforms
# import numpy as np
# import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../code') # little hack. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
from networks import FullyConnected, Conv
from verifier import analyze



# parse input
parser = argparse.ArgumentParser()
parser.add_argument('--net',
                    type=str,
                    choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                    required=True,
                    help='Neural network to refine maybe_robust for.')
parser.add_argument('--num',
                    type=int,
                    default=None,
                    help='Number of maybe_robust test cases to try to verify. (default: all)')
parser.add_argument('--timeout',
                    type=int,
                    default=3,
                    help='Timeout to use for the verifier, in seconds. (default: 3)')
parser.add_argument('--myverbose', 
                    action='store_true', # False by default
                    help="Run the analyzer verbosely.")
args = parser.parse_args()

# define "globals"
NET_NAME = args.net
BASE_DIR_PATH = '../test_cases_generated/' + args.net
# BASE_DIR_PATH = '../test_cases/' + args.net
NUM_TO_TRY = args.num # if None, try to verify all
TIMEOUT = args.timeout
VERBOSE = args.myverbose

DEVICE = 'cpu'
INPUT_SIZE = 28

# create output dir if it doesn't exist yet
os.makedirs(BASE_DIR_PATH+"/verifiable", exist_ok=True)



def main():
    print("Refining (up to) {} maybe_robust test cases \n \
            for network: {} \n \
            with timeout: {} \n \
            Executing analyzer verbosely: {}"
        .format(NUM_EXAMPLES_TO_GENERATE, NET_NAME, TIMEOUT, VERBOSE))

    # load concrete network
    net = get_net(NET_NAME)
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % NET_NAME, map_location=torch.device(DEVICE)))
    for p in net.parameters():
        p.requires_grad = False
    
    tried = 0
    with os.scandir(BASE_DIR_PATH+"/maybe_robust") as it:
        for f_name in it:
            inputs, eps, true_label, maybe_robustness = read_from_file(f_name.path)
            assert maybe_robustness == True
            
            # https://stackoverflow.com/questions/492519/timeout-on-a-function-call
            # not as precise and reliable as timeout-ing in the shell, only runs on Unix, doesn't work for multiple threads; 
            #   but does the job in this case, since we don't need precision.
            def timeout_handler(signum, frame):
                raise Exception("timed out")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(TIMEOUT)
            try:
                verified = analyze(net, inputs, eps, true_label, verbose=VERBOSE, net_name=NET_NAME)
            except Exception as exc:
                print(exc, "(not verified)")
                verified = False
            signal.alarm(0)

            if verified:
                # move f_name to BASE_DIR_PATH+"/verifiable"
                print("verified")
                src_path = f_name.path
                dst_path = os.path.join(BASE_DIR_PATH, "verifiable", f_name.name)
                print("moving {} to {}".format(src_path, dst_path))
                os.replace(src_path, dst_path)
            else:
                # else, leave it where it is
                print("failed to verify this test case :/")

            tried += 1
            if tried == NUM_TO_TRY:
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



if __name__ == "__main__":
    main()
