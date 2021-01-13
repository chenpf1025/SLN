import os
import time
import math
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from utils import train_noise, test, get_output, WeightEMA
import networks.resnet as resnet
from dataset import Clothing1M

def log(path, str):
    print(str)
    with open(path, 'a') as file:
        file.write(str)
 
# Settings
parser = argparse.ArgumentParser(description='PyTorch Clothing1M')
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_per_class', type=int, default=18976, help='num samples per class, if -1, use all samples.')
parser.add_argument('--datapath', default='./data/Clothing1M/', type=str, help='path to dataset')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--sigma', type=float, default=0.2, help='STD of Gaussian noise')
parser.add_argument('--correction', type=int, default=1, help='correction start epoch')
parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
args = parser.parse_args()

exp_name = 'clothing1m_sigma{:.1f}_batch{}_seed{}'.format(args.sigma, args.batch_size, args.seed)

if args.num_per_class>0:
    exp_name = 'num{}_'.format(args.num_per_class) + exp_name
    
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
kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
train_transform = transforms.Compose([transforms.Resize((256)),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
                                     ])
test_transform = transforms.Compose([transforms.Resize((256)),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
                                    ])

val_dataset = Clothing1M(args.datapath, mode='val', transform=test_transform)
test_dataset = Clothing1M(args.datapath, mode='test', transform=test_transform)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4*args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4*args.batch_size, shuffle=False, **kwargs)



# model
def create_model(ema=False):
    net = resnet.resnet50(pretrained=True)
    net.fc = torch.nn.Linear(net.fc.in_features, args.num_class)
    net = net.cuda()

    if ema:
        for param in net.parameters():
            param.detach_()
    return net
    
def learning_rate(lr_init, epoch):
    optim_factor = 0
    if(epoch > 5):
        optim_factor = 1
    return lr_init*math.pow(0.1, optim_factor)


net = create_model()
ema_net = create_model(ema=True)

cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
ema_optimizer = WeightEMA(net, ema_net)


# Training
val_best, test_at_best = 0, 0
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train_dataset = Clothing1M(args.datapath, mode='train', transform=train_transform, num_per_class=args.num_per_class)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
 
    lr = learning_rate(args.lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    # label-correction
    if epoch > args.correction:
        train_eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, **kwargs)
        noisy_targets = np.eye(args.num_class)[train_dataset.targets]
        
        log(logpath, 'Updating labels.\n')
        args.sigma = 0
        output, losses = get_output(ema_net, device, train_eval_loader)
        losses = (losses-min(losses))/(max(losses)-min(losses)) # normalize to [0,1]
        losses = losses.reshape([len(losses), 1])
      
        targets = losses*noisy_targets + (1-losses)*output
        train_dataset.targets = targets
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        
    _, train_acc = train_noise(args, net, device, train_loader, optimizer, epoch, ema_optimizer)
    _, test_acc_NoEMA = test(args, net, device, test_loader)
    _, val_acc = test(args, ema_net, device, val_loader)
    _, test_acc = test(args, ema_net, device, test_loader)
    

    
    if val_acc > val_best:
        val_best = val_acc
        test_at_best = test_acc
        torch.save(net.state_dict(), '{}/net.pth'.format(exp_name))
        torch.save(ema_net.state_dict(), '{}/ema_net.pth'.format(exp_name))
        
    log(logpath, 'Epoch: {}/{}, Time: {:.1f}s. '.format(epoch, args.epochs, time.time()-t0))
    log(logpath, 'Train: {:.2f}%, Test_NoEMA: {:.2f}%, Val: {:.2f}%, Test: {:.2f}%, Test_at_val_best: {:.2f}%.\n'.format(100*train_acc, 100*test_acc_NoEMA, 100*val_acc, 100*test_acc, 100*test_at_best))

