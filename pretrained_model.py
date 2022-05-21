import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from wideresnet import WideResNet
import helper

import numpy as np
import scipy.linalg

import pickle


# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, model, criterion, get_channels, encoding_dict, test_time, length):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
      if test_time and i == length:
        break 

      target = target.cuda(non_blocking=True)
      input = input.cuda(non_blocking=True)

      # compute output
      with torch.no_grad():
          output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1 = accuracy(output.data, target, topk=(1,))[0]
      losses.update(loss.data.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

    '''
    ADDED CODE FOR PROJECT
    '''
      if get_channels: # Used to get channel numbers, only ran once
        for weight in helper.weights:
          helper.num_channels.append(weight.size(dim=1))
          helper.feature_sizes.append(weight.size(dim=2))
        break
      else: 
        y = np.zeros((helper.m, 1))

        # Feature Encoding in which h = P*v
        # Feature Bundling in which y = bundling(h)
        for feature_size, weight, P in zip(helper.feature_sizes, helper.weights, helper.P_matrices):
          pooling = nn.MaxPool2d(feature_size, stride=1)
          v = pooling(weight)
          v = torch.reshape(v, (1, v.size(dim=1)))
          v = v.cpu().detach().numpy()
          v = np.transpose(v)
          h = np.dot(P, v)
          y = np.add(h, y) # y = bundling(h)

        if target.item() not in encoding_dict or test_time:
          encoding_dict[target.item()].append(y) # If image is being encoded for test set, just append
        else: # Else, if creating class descriptor, bundle all Hypervectors in same class
          encoding_dict[target.item()][0] = np.add(y, encoding_dict[target.item()][0])
        
      helper.weights.clear() 

      if i % 200 == 0: # For progress tracking 
        print(i)
      
    return top1.avg

def create_P(): 
    for c in helper.num_channels: # For each layer 
        a = np.random.random(size=(helper.m, c)) # Create random matrix m x c
        P = scipy.linalg.orth(a) # Make is semi-orthogonal 
        helper.P_matrices.append(P) 


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

# global args, best_prec1
args = parser.parse_args()

# Data loading code
normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

kwargs = {'num_workers': 1, 'pin_memory': True}

dataset_name = 'cifar10'
num_classes = 10

# Used for ID_encodings 
val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[dataset_name.upper()]('../data', train=False, download=True, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()

# create model
model = WideResNet(args.layers, num_classes,
                            args.widen_factor, dropRate=args.droprate)


# for training on multiple GPUs.
# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
# model = torch.nn.DataParallel(model).cuda()
model = model.cuda()

# Load model weights 
filepath = "drive/MyDrive/runs/WideResNet-28-10/model_best.pth.tar"
if os.path.isfile(filepath):
    print("=> loading best model weights '{}'".format(filepath))
    best_model = torch.load(filepath)
    model.load_state_dict(best_model['state_dict'])
    print("=> loaded best model weights")

# Gathers how many channels are in each layer 
temp = validate(val_loader, model, criterion, True, {}, False, 2)
print(helper.num_channels)

# Creates P matrix
create_P()
print(len(helper.P_matrices))

# Cut down number of layers used, for computational reasons
# Can increase or decrease by changing num_layers in helper.py 
helper.P_matrices = helper.P_matrices[:helper.num_layers]
helper.feature_sizes = helper.feature_sizes[:helper.num_layers]
helper.num_channels = helper.num_channels[:helper.num_layers]

'''
CIFAR10 - class descriptor 
'''
transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])

'''
If accuracy is low, try commenting out the "transform_train"
above and use the "transform_train" below, since the one
below was the transformations used to train the actual model 
'''
# transform_train = transforms.Compose([
#           transforms.ToTensor(),
#           transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
#                               (4,4,4,4),mode='reflect').squeeze()),
#             transforms.ToPILImage(),
#             transforms.RandomCrop(32),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#             ])

train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[dataset_name.upper()]('../data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)

print("Creating class_descriptors")
prec1 = validate(train_loader, model, criterion, False, helper.class_descriptors, False, 0)

'''
CIFAR10 - ID
'''

print("Creating ID encodings")
prec1 = validate(val_loader, model, criterion, False, helper.ID_encodings, True, helper.ID_length)


'''
CIFAR100 - Near_OOD
'''
dataset_name = 'cifar100'
cifar100 = datasets.__dict__[dataset_name.upper()]('../data', train=False, download=True, transform=transform_test)
targets = [i for i in range(10)]
indices = [i for i, label in enumerate(cifar100.targets) if label in targets]
cifar100_sub = torch.utils.data.Subset(cifar100, indices)

val_loader = torch.utils.data.DataLoader(
        cifar100_sub,
        batch_size=args.batch_size, shuffle=True, **kwargs)

print("Creating Near_OOD encodings")
prec1 = validate(val_loader, model, criterion, False, helper.near_OOD_encodings, True, helper.OOD_length)


# '''
# MNIST - Far_OOD
# '''
dataset_name = 'MNIST'

transform_test = transforms.Compose([
        transforms.Resize(size=(32,32)), 
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
        ])

val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[dataset_name.upper()]('../data', train=False, download=True, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, **kwargs)

print("Creating Far_OOD encodings")
prec1 = validate(val_loader, model, criterion, False, helper.far_OOD_encodings, True, helper.OOD_length)


# Print information 
for key, val in helper.class_descriptors.items():
    print(key, " ", len(val))
print()

for key, val in helper.ID_encodings.items():
    print(key, " ", len(val))
print()

for key, val in helper.near_OOD_encodings.items():
    print(key, " ", len(val))
print()

for key, val in helper.far_OOD_encodings.items():
    print(key, " ", len(val))
print()

# Save encodings
with open('all_encodings.pkl', 'wb') as f:
    pickle.dump([helper.class_descriptors, helper.ID_encodings, 
        helper.near_OOD_encodings, helper.far_OOD_encodings], f)


