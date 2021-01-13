import torch
import torch.nn.functional as F
import numpy as np

""" Training/testing """
# training
def train_noise(args, model, device, loader, optimizer, epoch, ema_optimizer=None):
    model.train()
    train_loss = 0
    correct = 0
     
    for data, target in loader:
        
        if len(target.size())==1:
            target = torch.zeros(target.size(0), args.num_class).scatter_(1, target.view(-1,1), 1) # convert label to one-hot

        data, target = data.to(device), target.to(device)
            
        # SLN
        if args.sigma>0:
            target += args.sigma*torch.randn(target.size()).to(device)
        
        output = model(data)
        loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1)*target, dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_optimizer:
            ema_optimizer.step()
        
        train_loss += data.size(0)*loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        if len(target.size())==2: # soft target
            target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    return train_loss/len(loader.dataset), correct/len(loader.dataset)


# testing
def test(args, model, device, loader, top5=False, criterion=F.cross_entropy):
    model.eval()
    test_loss = 0
    correct = 0
    correct_k = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if top5:
                _, pred = output.topk(5, 1, True, True)
                correct_k += pred.eq(target.view(-1,1)).sum().item()
    if top5:
        return test_loss/len(loader.dataset), correct_k/len(loader.dataset)
    else:
        return test_loss/len(loader.dataset), correct/len(loader.dataset)
 

def get_output(model, device, loader):
    softmax_outputs = []
    losses = []
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if len(target.size())==1:
                loss = F.cross_entropy(output, target, reduction='none')
            else:
                loss = -torch.sum(F.log_softmax(output, dim=1)*target, dim=1)
            output = F.softmax(output, dim=1)
               
            losses.append(loss.cpu().numpy())
            softmax_outputs.append(output.cpu().numpy())
            
    return np.concatenate(softmax_outputs), np.concatenate(losses)



# get hidden features (before the final fully-connected layer)
def get_feat(model, device, loader):
    feats = []
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            feat = model(data, get_feat=True)
            feats.append(feat.cpu().numpy())
    return np.concatenate(feats)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        #self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):                   
            # fix the error 'RuntimeError: result type Float can't be cast to the desired output type Long'
            #print(param.type())
            if param.type()=='torch.cuda.LongTensor':
                ema_param = param
            else:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            #param.mul_(1 - self.wd)