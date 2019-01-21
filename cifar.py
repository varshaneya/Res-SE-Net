'''
Training script for CIFAR-10/100

'''
from __future__ import print_function
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models.cifar as models
from utils import AverageMeter, accuracy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',#default 100
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('--depth', type=int, default=20, help='Model depth.')

# Miscs

parser.add_argument('--rounds', type=int, help='no of rounds to run')
parser.add_argument('--saveModel', type=str, help='no of rounds to run')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='prints out loss at each epoch')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed is set and experiments are conducted on this. Dont change.
manualSeed = 12
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)

best_acc = 0  # best test accuracy

def main(rnd):

    if os.path.exists('checkpoints') == False:
        os.mkdir('checkpoints')

    global best_acc
    start_epoch = 0

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        root = './CIFAR10'
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        root = './CIFAR100'
        num_classes = 100


    trainset = dataloader(root=root, train=True, download=True,  transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=root, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    elif args.arch.endswith('resenet'):#SE blocks only on bridge after downsample
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    elif args.arch.endswith('seresnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )

    if os.path.exists(os.path.join('checkpoints',args.arch)) == False:
        os.mkdir(os.path.join('checkpoints',args.arch))

    if args.evaluate:
        model.load_state_dict(torch.load(args.checkpoint)['model']).cuda()
    else:
        model = model.cuda()
		
    cudnn.benchmark = True
	
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.evaluate:
        print('\nEvaluation only')
        top1, top5 = test(testloader, model, criterion, start_epoch, use_cuda)
        print('Test accuracy:\nTop1 {}\nTop5 {}'.format(top1,top5))
        return

    # Train and val
	
    train_time_epoch = 0
    train_loss_epoch = []
    best_acc_top1 = 0
    best_acc_top5 = 0
    best_acc_epoch = 0
    bestModel = None

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        t = -time.time()
        train_loss = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        t += time.time()
        top1,top5 = test(testloader, model, criterion, epoch, use_cuda)
        if args.verbose:
            print('\nEpoch: [%d | %d] train loss: %f top1: %f top5: %f' % (epoch + 1, args.epochs, train_loss,top1,top5))
        if top1 > best_acc_top1:
            best_acc_top1 = top1
            best_acc_top5 = top5
            best_acc_epoch = epoch
            bestModel = model
        train_loss_epoch.append(train_loss)
        train_time_epoch += t

    print('Best accuracy\nTop1: {}\nTop5: {}\nAt epoch: {}'.format(best_acc_top1,best_acc_top5,best_acc_epoch))
    print('Average train loss is {}'.format(sum(train_loss_epoch)/len(train_loss_epoch)))
    save_checkpoint({'model':bestModel.state_dict(),'epoch':best_acc_epoch,'top1':best_acc_top1,'top5':best_acc_top5,
     'train_loss':train_loss_epoch,'optimizer':optimizer.state_dict()},
          checkpoint=args.checkpoint,filename=args.saveModel+str(rnd)+'.pth')
    return best_acc_top1,best_acc_top5,train_time_epoch/args.epochs


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    if args.arch.endswith('resnetdropskip'):
        model.isTrain = True #this introduces dropout to all the skip connections
    losses = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
   
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(losses)/len(losses)

def test(testloader, model, criterion, epoch, use_cuda):
    # switch to evaluate mode
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
    return top1.avg,top5.avg

def save_checkpoint(state, checkpoint, filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    top1 = []
    top5 = []
    train_time = []
    for rnd in range(args.rounds):
        state['lr'] = args.lr
        print('Round {}:\n'.format(rnd+1))
        output = main(rnd+1)
        if args.evaluate:
            break
        else:
            test_loss_top1,test_loss_top5,train_time_epoch = output
        top1.append(test_loss_top1)
        top5.append(test_loss_top5)
        train_time.append(train_time_epoch)
    print('Average train time per epoch is {}'.format(sum(train_time)/len(train_time)))
    print('Average best test accuracy:\nTop1 {}\nTop5 {}'.format(sum(top1)/len(top1),sum(top5)/len(top5)))
