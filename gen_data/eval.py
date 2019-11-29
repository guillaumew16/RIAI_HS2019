import argparse
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from utils import Net, Normalize
from attacks import pgd_untargeted


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default='42', help='seed')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='device')
parser.add_argument('--defense', type=str, choices=['True', 'False'], default='False', help='defense')
parser.add_argument('--num_epochs', type=int, default=1, help='epochs')
args = parser.parse_args()

# Setting the random number generator
torch.manual_seed(args.seed)

# Setting up the Model
model = nn.Sequential(Normalize(), Net())
model.load_state_dict(torch.load(f'./models/Net_{args.num_epochs}_{args.defense}'))
model.to(args.device)
model.eval()

# disable batches
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)


# number of samples we evaluate on
num_samples = 1000

xlist = [] # collect x tensors to use the same values in the next loop
ylist = [] # collect y tensors to use the same values in the next loop

# not attacked
num_correct = 0
for idx, (x, y) in enumerate(tqdm(itertools.islice(train_loader, num_samples))):
    xlist.append(x)
    ylist.append(y)
    x, y = x.to(args.device), y.to(args.device)
    out = model(x)
    pred = torch.max(out, dim=1)[1]
    num_correct += pred.eq(y).item()

print('Accuracy %i samples original %.5lf' % (num_samples, num_correct / num_samples))

# attacked
num_correct = 0
for cnt in tqdm(range(num_samples)):
    # retrieve the same point as before
    x = xlist[cnt].to(args.device)
    y = ylist[cnt].to(args.device)
    
    # perturb the point using a PGD attack
    x = pgd_untargeted(model, x, y, 5, 0.08, 0.05, args.device)
    
    out = model(x)
    pred = torch.max(out, dim=1)[1]
    num_correct += pred.eq(y).item()

print('Accuracy %i samples perturbed %.5lf' % (num_samples, num_correct / num_samples))