# Quick and dirty test to show that art.utils.load_mnist is bad
use_ART_mnist = True




import argparse
import torch
from torchvision import datasets, transforms
import numpy as np

import sys
sys.path.insert(1, '../code') # little hack. https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
from networks import FullyConnected, Conv

from art.data_generators import PyTorchDataGenerator
from art.utils import load_mnist, to_categorical

DEVICE = 'cpu'
INPUT_SIZE = 28

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

net = get_net(args.net)
net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

DATASET_PATH = '../mnist_data'
mnist_dataset = datasets.MNIST(DATASET_PATH, train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))
mnist_loader = torch.utils.data.DataLoader(mnist_dataset, shuffle=True, batch_size=1000)

# use_ART_mnist = False # <-- set this at top of file
if use_ART_mnist:
    # # Load the MNIST dataset
    # (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist() # <--- tests indicate that art.utils.load_mnist is not what we want
    # # Swap axes to PyTorch's NCHW format
    # x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
    # x_test = torch.from_numpy(x_test)
    artDataGenerator = PyTorchDataGenerator(mnist_loader, size=None, batch_size=1000)
    x_test, y_test = artDataGenerator.get_batch()
    x_test = torch.from_numpy(x_test)
    y_test = to_categorical(y_test, 10)
else:
    for idx, (x, true_label) in enumerate(mnist_loader):
        x_test = x
        y_test = to_categorical(true_label.numpy(), 10)
        break
print(x_test.shape)
print(y_test.shape)
print(type(x_test))
print(type(y_test))


# Evaluate the ART classifier on benign test examples (sanity check)
for p in net.parameters():
    p.requires_grad = False
predictions = net(x_test)
predictions = predictions.numpy()
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print('Accuracy on benign test examples: {}%'.format(accuracy * 100))

