import torch
from torchvision import datasets, transforms
import argparse
import os
import random
from adv import pgd_untargeted

DEVICE = 'cpu'
INPUT_SIZE = 28

parser = argparse.ArgumentParser()
parser.add_argument('--net',
                    type=str,
                    choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                    required=True,
                    help='Neural network to generate test cases for.')
args = parser.parse_args()

BASE_DIR_PATH = '../test_cases_generated/' + args.net

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
mnist_loader = torch.utils.data.DataLoader(dataset, shuffle=True)

def generate_uid():
    uids = []
    with os.scandir(BASE_DIR_PATH) as it:
        for f_name in it:
            under_score_pos = f_name.rfind('_')
            if under_score_pos == -1:
                raise Warning(BASE_DIR_PATH+" contains a file with bad filename: "+f_name+" (filename should contain '_')")
                continue
            uids.append( f_name[0:under_score_pos] )
    idx = 0
    while True:
        uid = "img" + str(idx) # candidate uid
        if not uid in uids:
            yield uid
        idx += 1

def write_to_file(x, eps, true_label, robust):
    # x: torch.Tensor
    filename = generate_uid() + "_" + str(eps) + ".txt"
    robustness_path = 'maybe_robust' if robust else 'not_robust'
    filename = os.path.join(BASE_DIR_PATH, robustness_path, filename)
    print("Writing data to file", filename)
    with open(filename, 'w') as f:
        f.write(str(true_label) + "\n")
        # TODO: write x, one scalar by row
    print("Finished writing.")

# for testing purposes (check that write_to_file then read_from_file returns the original tensor)
def read_from_file(filename):
    robust = None
    if filename.find('maybe_robust'):
        robust = True
    else if filename.find('not_robust'):
        robust = False
    if robust == None:
        raise ValueError("bad filename: " + str(filename))
    with open(filename, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(filename[:-4].split('/')[-1].split('_')[-1])
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    return inputs, eps, true_label, robust



for idx, (x, true_label) in enumerate(train_loader): # batch_size=1 by default
    eps = random.uniform(0.005, 0.2) # eps ranges between 0.005 and 2
    eps_step = eps/10 # step size in FGSM (see adv.py)    
    # look for an adversarial example
    robust = True
    for _ in range(10): # try multiple times (adv.pgd_ includes some randomness)
        k = 10 # nb of projections in PGD
        x_adv = pgd_untargeted(net, x, true_label, k, eps, eps_step, device='cpu', clip_min=0, clip_max=1) # candidate adversarial example
        outs = net(x_adv)
        pred_label = outs.max(dim=1)[1].item()
        if pred_label != true_label:
            robust = False
            break
    write_to_file(x, eps, true_label, robust)


