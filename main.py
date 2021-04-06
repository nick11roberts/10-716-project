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
from torch.utils.tensorboard import SummaryWriter
from auto_augment import AutoAugment, Cutout
from archive import autoaug_paper_cifar10
from FastAutoAugment.data import Augmentation
from models import S_FC as Net

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_nonzero_parameters(model):
    return sum(
        torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)

def get_images(w, n=1):
    select = torch.randint(0, w.shape[0], size=(n,))
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
        scheduler.step()

        cur_step = (len(train_loader) * (epoch-1)) + batch_idx
        w = model.fc1.weight.data
        images = get_images(w, 32)
        grid = make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_scalar('Loss/train', loss.item(), cur_step)
        writer.add_scalar('Metadata/learning_rate', 
            scheduler.get_last_lr()[0], cur_step)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, epoch, writer):
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

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    writer.add_scalar('Loss/test', test_loss, epoch-1)
    writer.add_scalar('Accuracy/test', acc, epoch-1)
    writer.add_scalar('Parameters/nnz', 
        count_nonzero_parameters(model), epoch-1)
    writer.add_scalar('Parameters/sparsity', 
        count_nonzero_parameters(model) / count_parameters(model), epoch-1)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), 100. * acc))


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
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    writer = SummaryWriter()

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    augs = [
        Augmentation(autoaug_paper_cifar10())
        ]
    normalizer = [
        transforms.ToTensor(), # TODO need this?
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
        ]
    transform = transforms.Compose(augs + normalizer)
    test_transform = transforms.Compose(normalizer)
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False,
                       transform=test_transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    print('Num params:', count_parameters(model))

    # TODO parameterize this
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = BetaLasso(model.parameters(), lr=args.lr, reg=1e-06, beta=50)

    scheduler = CosineAnnealingLR(optimizer, 
        T_max=args.epochs*len(train_loader))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, scheduler, epoch, 
            writer)
        test(model, device, test_loader, epoch, writer)
        writer.flush()

    if args.save_model:
        # TODO give model a better name
        torch.save(model.state_dict(), "cifar10_net.pt")


if __name__ == '__main__':
    main()