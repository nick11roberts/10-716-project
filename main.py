from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from betalasso import L1, Lasso, BetaLasso
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
from auto_augment import AutoAugment, Cutout
from theconf import Config as C
from archive import autoaug_paper_cifar10
from FastAutoAugment.data import Augmentation, get_dataloaders
from models import S_FC as Net

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def count_parameters(model, layer=''):
    if layer == '':
        l = model.parameters()
    else:
        l = [param for name, param in model.named_parameters() 
            if layer in name]
    return sum(p.numel() for p in l if p.requires_grad)

def count_nonzero_parameters(model, layer=''):
    if layer == '':
        l = model.parameters()
    else:
        l = [param for name, param in model.named_parameters() 
            if layer in name]
    return sum(
        torch.count_nonzero(p) for p in l if p.requires_grad)

def get_images(w, n=1):
    select = torch.randint(0, w.shape[0], size=(n,))
    #select = torch.arange(n)
    w_im = w[select].unflatten(1, (3, 32, 32))
    return w_im

def train(args, model, device, train_loader, optimizer, scheduler, 
        epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(data)
            loss = F.nll_loss(output, target)
        #loss.backward()
        #optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(epoch - 1 + float(batch_idx + 1) / len(train_loader))

        cur_step = (len(train_loader) * (epoch-1)) + batch_idx
        writer.add_scalar('Loss/train', loss.item(), cur_step)
        writer.add_scalar('Metadata/learning_rate', 
            scheduler.get_last_lr()[0], cur_step)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, epoch, writer, split='test'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(data)
                test_loss += F.nll_loss(output, target, 
                    reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, 
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    n = len(test_loader.sampler)
    test_loss /= n
    acc = correct / n

    w = model.layer1.weight.data
    images = get_images(w, 32)
    grid = make_grid(images)
    writer.add_image('images', grid, epoch-1)
    writer.add_scalar(f'Loss/{split}', test_loss, epoch-1)
    writer.add_scalar(f'Accuracy/{split}', acc, epoch-1)
    writer.add_scalar('Parameters/nnz', 
        count_nonzero_parameters(model), epoch-1)
    writer.add_scalar('Parameters/sparsity', 
        count_nonzero_parameters(model) / count_parameters(model), epoch-1)
    writer.add_scalar('Parameters/nnz_first_layer', 
        count_nonzero_parameters(model, layer='layer1'), epoch-1)
    writer.add_scalar('Parameters/sparsity_first_layer', 
        count_nonzero_parameters(model, layer='layer1') / count_parameters(
            model, layer='layer1'), epoch-1)
    writer.add_scalar('Parameters/nnz_second_layer', 
        count_nonzero_parameters(model, layer='fc2'), epoch-1)
    writer.add_scalar('Parameters/sparsity_second_layer', 
        count_nonzero_parameters(model, layer='fc2') / count_parameters(
            model, layer='fc2'), epoch-1)
    writer.add_scalar('Parameters/nnz_last_layer', 
        count_nonzero_parameters(model, layer='fc3'), epoch-1)
    writer.add_scalar('Parameters/sparsity_last_layer', 
        count_nonzero_parameters(model, layer='fc3') / count_parameters(
            model, layer='fc3'), epoch-1)

    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, test_loss, correct, n, 100. * acc))

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Learning convolutions from scratch example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size', type=int, default=1000, 
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between log events')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    # Hyperparameters
    parser.add_argument('--reg1', type=float, default=2e-5, 
                        metavar='REG1', # TODO set to best hp
                        help='first layer lambda (default: 2e-05)')
    parser.add_argument('--reg', type=float, default=5e-6, 
                        metavar='REG', # TODO set to best hp
                        help='lambda (default: 5e-6)')
    parser.add_argument('--beta', type=float, default=50, 
                        metavar='BETA',
                        help='beta (default: 50)')
    parser.add_argument('--dropout', action='store_true', default=False,
                        help='Use dropout on the last two layers') 
    parser.add_argument('--full-train', action='store_true', default=False,
                        help='Train using the full training set') 
                        # TODO set to best hp
    # TODO parameterize SGD hps etc
    # TODO parameterize which model to use etc
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    writer = SummaryWriter(
        comment=f'__reg1{args.reg1}_reg{args.reg}_dropout{args.dropout}__')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_sampler, train_loader, valid_loader, test_loader = get_dataloaders(
        'cifar10', args.batch_size, '../data', 
        split=0.0 if args.full_train else 0.15)

    model = Net(dropout=args.dropout).to(device)
    print('Num params:', count_parameters(model))
    print(count_parameters(model, layer='layer1'))
    print(count_parameters(model, layer='fc2'))
    print(count_parameters(model, layer='fc3'))

    # TODO parameterize this
    first_layer = [param for name, param in model.named_parameters()
                        if 'layer1' in name]
    other_layers = [param for name, param in model.named_parameters()
                        if 'layer1' not in name]
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = BetaLasso([
            {'params': first_layer, 
                'lr': args.lr, 'reg': args.reg1, 'beta': args.beta},
            {'params': other_layers, 
                'lr': args.lr, 'reg': args.reg, 'beta': args.beta},])

    #scheduler = CosineAnnealingLR(optimizer, 
    #    T_max=args.epochs*len(train_loader))
    # TODO trying cosine scheduler settings from fast autoaugment
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.)
    if False and C.get()['lr_schedule'].get(
        'warmup', None) and C.get()['lr_schedule']['warmup']['epoch'] > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=C.get()['lr_schedule']['warmup']['multiplier'],
            total_epoch=C.get()['lr_schedule']['warmup']['epoch'],
            after_scheduler=scheduler
        )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, 
            optimizer, scheduler, epoch, writer)
        if not args.full_train:
            test(model, device, valid_loader, epoch, writer, split='valid')
        test(model, device, test_loader, epoch, writer, split='test')
        writer.flush()

    if args.save_model:
        # TODO give model a better name?
        torch.save(model.state_dict(), 
        f'models/__reg1{args.reg1}_reg{args.reg}_dropout{args.dropout}__.pt')


if __name__ == '__main__':
    _ = C('s_fc_cifar10.yaml')
    main()