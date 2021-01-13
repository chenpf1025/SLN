import os
import time
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn


from utils import train_noise, test, get_output, WeightEMA
from dataset import get_cifar_dataset
from networks.wideresnet import Wide_ResNet

def log(path, str):
    print(str)
    with open(path, 'a') as file:
        file.write(str)
 
# Settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--datapath', default='./data/CIFAR10', type=str, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--noise_mode', type=str, default='dependent', help='Noise mode')
parser.add_argument('--noise_rate', type=float, default=0.4, help='Noise rate')
parser.add_argument('--sigma', type=float, default=0.5, help='STD of Gaussian noise')
parser.add_argument('--correction', type=int, default=250, help='correction start epoch')
parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
args = parser.parse_args()


exp_name = 'sigma{:.1f}_{}_{}{:.1f}_seed{}'.format(args.sigma, args.dataset, args.noise_mode, args.noise_rate, args.seed) 


if 0<args.correction<args.epochs:
    exp_name = 'correction_' + exp_name
else:
    args.correction = args.epochs

    
if not os.path.exists(exp_name):
    os.mkdir(exp_name)
logpath = '{}/log.txt'.format(exp_name)
log(logpath, 'Settings: {}\n'.format(args))

device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(args.gpu_id)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# dataset
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dataset, test_dataset = get_cifar_dataset(args.dataset, args.datapath, args.noise_mode, args.noise_rate)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
train_eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

noisy_targets = train_dataset.targets
noisy_targets = np.eye(args.num_class)[noisy_targets] # to one-hot


# model
net = Wide_ResNet(num_classes=args.num_class).cuda()
ema_net = Wide_ResNet(num_classes=args.num_class).cuda()
for param in ema_net.parameters():
    param.detach_()
    
cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
ema_optimizer = WeightEMA(net, ema_net)


# Training
global_t0 = time.time()
for epoch in range(1, args.epochs + 1):
    t0 = time.time()    
        
    # label-correction
    if epoch > args.correction:
        log(logpath, 'Updating labels.\n')
        args.sigma = 0
        output, losses = get_output(ema_net, device, train_eval_loader)
        output = np.eye(args.num_class)[output.argmax(axis=1)]
        losses = (losses-min(losses))/(max(losses)-min(losses)) # normalize to [0,1]
        losses = losses.reshape([len(losses), 1])

        targets = losses*noisy_targets + (1-losses)*output
        train_dataset.targets = targets
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        
    _, train_acc = train_noise(args, net, device, train_loader, optimizer, epoch, ema_optimizer)
    _, test_acc = test(args, ema_net, device, test_loader)
    _, test_acc_NoEMA = test(args, net, device, test_loader)


    log(logpath, 'Epoch: {}/{}, Time: {:.1f}s. '.format(epoch, args.epochs, time.time()-t0))
    log(logpath, 'Train: {:.2f}%, Test_NoEMA: {:.2f}%, Test: {:.2f}%.\n'.format(100*train_acc, 100*test_acc_NoEMA, 100*test_acc))
    
log(logpath, '\nTime: {:.1f}s.\n'.format(time.time()-global_t0))

# Saving
torch.save(net.state_dict(), '{}/net.pth'.format(exp_name))
torch.save(ema_net.state_dict(), '{}/ema_net.pth'.format(exp_name))