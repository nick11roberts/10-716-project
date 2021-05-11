from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
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
        epoch, writer, eps=1e-6):
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

        if args.prune_iterations > 1:
            # Freezing Pruned weights by making their gradients Zero
            for name, p in model.named_parameters():
                if 'weight' in name:
                    # Slow version. 
                    #tensor = p.data.cpu().numpy()
                    #grad_tensor = p.grad.data.cpu().numpy()
                    #grad_tensor = np.where(tensor < eps, 0, grad_tensor)
                    #p.grad.data = torch.from_numpy(grad_tensor).to(device)
                    
                    # My version TODO check this - this is maybe numerically unstable? 
                    #p.grad.mul_(p.abs() >= eps)

                    # Probably the right version
                    tensor = p.data
                    grad_tensor = p.grad
                    grad_tensor = torch.where(
                        tensor.abs() < eps, 
                        torch.zeros_like(grad_tensor), 
                        grad_tensor)
                    p.grad.data = grad_tensor

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

def run_train(args, model):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    
    if args.use_sgd:
        iden = f'__SGD__momentum{args.momentum}__wd{args.weight_decay}_dropout{args.dropout}__'
    else:
        iden = f'__reg1{args.reg1}_reg{args.reg}_dropout{args.dropout}__sti{args.soft_threshold_iters}__'
    
    writer = SummaryWriter(comment=iden)

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

    model = model.to(device)
    print('Num params:', count_parameters(model))
    print(count_parameters(model, layer='layer1'))
    print(count_parameters(model, layer='fc2'))
    print(count_parameters(model, layer='fc3'))

    if args.use_sgd:
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay)
    else:
        first_layer = [param for name, param in model.named_parameters()
                       if 'layer1' in name]
        other_layers = [param for name, param in model.named_parameters()
                        if 'layer1' not in name]
        optimizer = BetaLasso([
                {'params': first_layer, 
                    'lr': args.lr, 'reg': args.reg1, 'beta': args.beta,},
                {'params': other_layers, 
                    'lr': args.lr, 'reg': args.reg, 'beta': args.beta,},],
                soft_threshold_iters=args.soft_threshold_iters)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, 
            optimizer, scheduler, epoch, writer)
        if not args.full_train:
            test(model, device, valid_loader, epoch, writer, split='valid')
        test(model, device, test_loader, epoch, writer, split='test')
        writer.flush()

    if args.save_model:
        # TODO give model a better name?
        torch.save(model.state_dict(), f'models/' + iden + '.pt')

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
    parser.add_argument('--soft-threshold-iters', type=int, default=1, 
                        help='soft_threshold_iters (default: 1)')
    parser.add_argument('--dropout', action='store_true', default=False,
                        help='Use dropout on the last two layers')
    parser.add_argument('--full-train', action='store_true', default=False,
                        help='Train using the full training set') 
                        # TODO set to best hp
    # SGD hps etc
    parser.add_argument('--use-sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--momentum', type=float, default=0.0, 
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0, 
                        help='SGD weight decay')

    # LTH IMP stuff
    parser.add_argument('--prune-iterations', type=int, default=1, 
                        help='prune iterations (default: 1)')
    parser.add_argument('--prune-percent', type=int, default=20,
                        help='Percent to prune at each prune iteration')

    # NAS stuff
    parser.add_argument('--offline-eval', action='store_true', default=False,
                        help='Run offline evaluation')

    args = parser.parse_args()

    # TODO parameterize which model to use etc.
    global model
    model = Net(dropout=args.dropout)
    model.fc3.weight.output = True # Indicates half pruning rate
    initial_state_dict = copy.deepcopy(model.state_dict())  # save init
    make_mask(model) # initial mask

    # SUPERNET TRAINING
    for p_iter in range(args.prune_iterations):
        # Prune, re-initialize
        if p_iter != 0:
            prune_by_percentile(args.prune_percent)
            original_initialization(mask, initial_state_dict)
        run_train(args, model)

    # OFFLINE EVALUATION
    if args.offline_eval > 0:

        # Discretize supernet by setting mask, if it isn't already set
        if args.prune_iterations <= 1:
            prune_current_zeros()
        
        original_initialization(mask, initial_state_dict)

        # Set optimal SGD hyperparameters for offline eval
        args.epochs = 4000
        args.use_sgd = True
        args.momentum = 0.9
        #args.weight_decay = 0.0002 
        # Don't use weight decay - the architectures are already sparse
        args.dropout = False
        args.full_train = True
        args.save_model = True
        args.prune_iterations = 10 # To ensure that grad gets zeroed
        run_train(args, model)



def prune_current_zeros(eps=1e-6):
    global step
    global mask
    global model

    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            weight_dev = param.device
            new_mask = mask[step] * (np.abs(tensor) >= eps)
            print(new_mask)
            #new_mask = np.where(abs(tensor) < eps, 0, mask[step])
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0

# Prune by Percentile module
def prune_by_percentile(percent, resample=False, reinit=False, **kwargs):
    global step
    global mask
    global model

    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] 
            # flattened array of nonzero values
            if hasattr(param, 'output'):
                percentile_value = np.percentile(abs(alive), percent // 2)
            else:
                percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])
            
            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            mask[step] = new_mask
            step += 1
    step = 0

# Function to make an empty mask of the same size as the model
def make_mask(model):
    global step
    global mask
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            step = step + 1
    mask = [None]* step 
    step = 0
    for name, param in model.named_parameters(): 
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    step = 0

def original_initialization(mask_temp, initial_state_dict):
    global model
    
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0

if __name__ == '__main__':
    _ = C('s_fc_cifar10.yaml')
    main()