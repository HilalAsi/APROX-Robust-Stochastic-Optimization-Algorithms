'''Main trainining script'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import copy

import math
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler
import torch.utils.data

import os

import models
import datasets
import configs
from losses import *
import optimizers_pytorch
from torch.nn import CrossEntropyLoss
from configs import TrainConfig

from stability_check import *


def generate_data(cfg):
    dataset_train, dataset_val = eval("datasets." + cfg.dataset)(**cfg.dataset_cfg)
    dataset_sizes = {'train': len(dataset_train), 'val': len(dataset_val)}
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size= cfg.batch_size,
                                                   shuffle=True, num_workers= cfg.loader_workers)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size= cfg.batch_size,
                                                 shuffle=False, num_workers= cfg.loader_workers)

    def add_training_set(d):
        if "dataset" in d:
            d["dataset"] = dataset_train
        return d

    metric_funcs = {name: eval(class_name)(**add_training_set(kwargs)) for name, class_name, kwargs in cfg.metrics}

    return  train_loader, val_loader, dataset_sizes, metric_funcs


def create_net(cfg):
    # Model
    net = eval("models." + cfg.model)(**cfg.model_cfg)
    if cfg.use_cuda:
        net.cuda()
    # Optimizer
    if 'module' in cfg.optimizer_cfg:
        cfg.optimizer_cfg['module'] = net

    if cfg.model.startswith('pretrained'):
        #learn only fine-tunable parameters
        optimizer = eval(cfg.optimizer)(filter(lambda p: p.requires_grad, net.parameters()), **cfg.optimizer_cfg)
    else:
        optimizer = eval(cfg.optimizer)(net.parameters(), **cfg.optimizer_cfg)

    # LR Scheduler
    lr_scheduler = eval(cfg.lr_scheduler)(optimizer, **cfg.lr_scheduler_cfg)

    return net, optimizer, lr_scheduler

def create_config(model_name, dataset, optimizer_name,
                  dataset_size, loss, max_epoch, init_stepsize, batch_size,use_cuda ):
    cfg = TrainConfig
    cfg.model = model_name
    cfg.loader_workers = 2
    cfg.use_cuda = use_cuda

    if dataset == 'CIFAR10':
        cfg.model_cfg = {'in_channels': 3, 'out_channels': 10}
    elif dataset == 'MNIST':
        cfg.model_cfg = {'in_channels': 1, 'out_channels': 10}
    elif dataset == 'Stanford_dogs':
        cfg.model_cfg = {'in_channels': 3, 'out_channels': 120}

    if cfg.model.startswith('pretrained'):
        num_training_layers = 1
        if cfg.model[-1]>= '0' and cfg.model[-1]<='9':
            num_training_layers = int(cfg.model[-1])
            cfg.model = cfg.model[:-1]
        cfg.model_cfg['num_training_layers'] = num_training_layers

    cfg.epochs_max = max_epoch
    # define the dataset
    cfg.dataset = dataset
    if dataset_size > 0:
        cfg.dataset_cfg = {'truncate': int(dataset_size)}
    cfg.batch_size = batch_size
    cfg.optimizer = optimizer_name
    # modify optimizer and step size
    if optimizer_name == 'sgd':
        cfg.optimizer = 'torch.optim.SGD'
        cfg.optimizer_cfg = {'lr': init_stepsize}
    elif optimizer_name == 'sgd_momentum':
        cfg.optimizer = 'torch.optim.SGD'
        cfg.optimizer_cfg = {'lr': init_stepsize, 'momentum': 0.9}
    elif optimizer_name == 'sgd_momentum_wd':
        cfg.optimizer = 'torch.optim.SGD'
        cfg.optimizer_cfg = {'lr': init_stepsize, 'momentum': 0.9, 'weight_decay': 5e-4}
    elif optimizer_name == 'adam':
        cfg.optimizer = 'torch.optim.Adam'
        cfg.optimizer_cfg = {'lr': init_stepsize/100.0} #divide by 100 for adam
    elif  optimizer_name == 'truncated':
        cfg.optimizer = 'optimizers_pytorch.Truncated'
        cfg.optimizer_cfg = {'lr': init_stepsize}
    elif optimizer_name == 'trunc_adagrad':
        cfg.optimizer = 'optimizers_pytorch.TruncatedAdagrad'
        cfg.optimizer_cfg = {'lr': init_stepsize}
        lr_scheduler = None
    else:
        cfg.optimizer = optimizer_name
        cfg.optimizer_cfg = {'lr': init_stepsize, 'momentum': 0.9}

    # modify learning rate scheduler
    cfg.lr_scheduler = 'torch.optim.lr_scheduler.StepLR'
    cfg.lr_scheduler_cfg = {'step_size': 20, 'gamma': 0.1}

    # metrics
    cfg.metrics = [('loss', loss, {}),
               ('accuracy', 'TopKAccuracy', {'k': 1})]  #MultiLabelMarginLoss
    return cfg

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, use_cuda, num_epochs=50):

    val_losses = np.zeros(num_epochs)
    val_accuracies = np.zeros(num_epochs)
    train_losses = np.zeros(num_epochs)
    train_accuracies = np.zeros(num_epochs)

    best_acc = 0.0
    best_loss = -1.0
    epoch_loss = -1.0
    for epoch in range(num_epochs):
        #check for divergence
        if best_loss == 0 or np.isfinite(epoch_loss) is False:
            break

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('Best Loss: {:4f} Best val Acc: {:4f}'.format(best_loss, best_acc))
        # Each epoch has a training and validation phase
        for phase in ['val', 'train']:
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                if use_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, features = model(Variable(inputs))
                _, preds = torch.max(outputs, 1)

                def closure():
                    loss = criterion(outputs, Variable(labels))
                    loss.backward()
                    return loss

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss = optimizer.step(closure)
                else:
                    loss = criterion(outputs, Variable(labels))

                # statistics (use the next 2 lines if your version does not support this)
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += preds.eq(Variable(labels)).sum().data[0]

                # statistics
                #running_loss += loss.item() * inputs.size(0)
                #running_corrects += preds.eq(Variable(labels)).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 1.0 * running_corrects / dataset_sizes[phase]

            if phase == 'train':
                train_losses[epoch], train_accuracies[epoch] = epoch_loss, epoch_acc
            else:
                val_losses[epoch], val_accuracies[epoch] = epoch_loss, epoch_acc

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and (epoch_loss < best_loss or best_loss < 0):
                best_loss = epoch_loss
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

        #check for divergence
        if (np.isfinite(val_losses[epoch])==False) or (np.isfinite(train_losses[epoch])==False):
            break
        print()


    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))

    if epoch < num_epochs - 1:
        train_losses = train_losses[:epoch+1]
        train_accuracies = train_accuracies[:epoch+1]
        val_losses = val_losses[:epoch+1]
        val_accuracies = val_accuracies[:epoch+1]

    return train_losses, train_accuracies, val_losses, val_accuracies



# Main Optimization function. Take model parameters and returns train\validation losses and accuracies,
# which are computed every epoch.
#
# dataset - name of dataset from { 'MNIST', 'CIFAR10', 'Big_CIFAR10'. 'STL10', 'Hymenopetra'}
# optimizer_name - type of optimizer from {'torch.optim.SGD', 'optimizers.SGDWithTruncation', 'optimizers.prox_linear' }
# lr_scheduler - type to learning rate scheduler from {'stepLR', 'scheduler_sum_squares' }
# dataset_size - size of dataset
# max_epoch - maximum number of epochs
# init_stepsize - initial step size
# batch_size - batch size
# use_cuda - specifies if to use gpu
def NN_optimize_fast(model_name, dataset='CIFAR10', optimizer_name='sgd', dataset_size=500,
                     loss='nn.MultiMarginLoss' ,max_epoch=50, init_stepsize=0.1, batch_size=100, use_cuda=False):
    if use_cuda:
        use_cuda = torch.cuda.is_available()

    print('use cuda = ' + str(use_cuda))
    # define the configuration
    cfg = create_config(model_name, dataset, optimizer_name,
                        dataset_size, loss, max_epoch, init_stepsize, batch_size,use_cuda )

    # generate datasets
    train_loader, val_loader, dataset_sizes, metric_funcs = generate_data(cfg)

    # create the network
    net, optimizer, lr_scheduler = create_net(cfg)

    # GPU training
    if use_cuda:
        net.cuda()
        for f in metric_funcs.values():
            f.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    print('init_stepsize = ' + str(init_stepsize))
    dataloaders = {'train': train_loader , 'val': val_loader }
    loss_func = metric_funcs['loss']
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(net, loss_func,
                                                                             optimizer, lr_scheduler, dataloaders,
                                                                             dataset_sizes, use_cuda, max_epoch)
    return train_losses, train_accuracies, val_losses, val_accuracies




def get_label(opt):
    # possible optimizers are specific here.
    label = opt
    if opt == 'sgd_momentum':
        label = 'SGM_momentum'
    elif opt == 'sgd':
        label = 'SGM'
    elif opt == 'sgd_momentum_wd':
        label = 'SGM_momentum_wd'
    elif opt == 'truncated':
        label = 'Truncated'

    return label
