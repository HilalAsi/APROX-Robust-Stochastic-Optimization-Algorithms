from configs import TrainConfig
import re
from os.path import join as pjoin
import numpy as np


def apply_setting(full_cfg_str, **extra_cfgs):
    name_lst = []
    cfg_strs = re.split('[ +]+', full_cfg_str)
    for cfg_str in cfg_strs:
        split_cfg_str = re.split('[=,]+', cfg_str)
        command, args = split_cfg_str[0], split_cfg_str[1:]
        if command.startswith('**'): # starting command with ** hides it completely
            command = command[2:]
        elif command.startswith('*'):  # starting command with * keeps only command args
            command = command[1:]
            name_lst += [','.join(args)]
        else:
            args_str = '=' + ','.join(args) if len(args) > 0 else ''
            name_lst += [command + args_str]
        # print("command = %s, args = %s" % (command, args))
        cfg = eval(command)(*args)

    for k, v in extra_cfgs.items():
        setattr(cfg, k, v)
        name_lst += '%s=%s' % (k, v)

    cfg.name = '+'.join(name_lst)


# -------------


def VGG(size='11'):
    cfg = TrainConfig

    cfg.model = 'VGG'
    cfg.model_cfg = dict(vgg_name='VGG'+size)

    return cfg


def ResNet(order='18'):
    cfg = TrainConfig

    cfg.model = 'ResNet' + order
    cfg.model_cfg = {}

    return cfg


def LeNet():
    cfg = TrainConfig

    cfg.model = 'LeNet'
    cfg.model_cfg = {}

    return cfg


def truncate_dataset(truncate='500'):
    cfg = TrainConfig
    cfg.dataset = 'CIFAR10'
    cfg.dataset_cfg = {'truncate': int(truncate)}
    return cfg


def batch_size(size_train='100', size_val=None):
    cfg = TrainConfig
    cfg.batch_size = int(size_train)
    if size_val:
        cfg.batch_size_val = int(size_val)
    return cfg


def epochs(num):
    TrainConfig.epochs_max = int(num)
    return TrainConfig


def dropout(d='0.5'):
    TrainConfig.model_cfg['dropout'] = float(d)
    return TrainConfig


def adam():
    cfg = TrainConfig

    cfg.optimizer = 'torch.optim.Adam'
    cfg.optimizer_cfg = {'lr': 0.001}

    cfg.lr_scheduler = 'torch.optim.lr_scheduler.MultiStepLR'
    cfg.lr_scheduler_cfg = {'milestones': [int(0.6 * cfg.epochs_max), int(0.8 * cfg.epochs_max)],
                            'gamma': 0.2}

    return cfg


def sgd():
    cfg = TrainConfig

    cfg.optimizer = 'torch.optim.SGD'
    cfg.optimizer_cfg = {'lr': 0.1, 'momentum': 0.9}

    if cfg.metrics[0][1] in ('WeightedRegressionLoss', 'SampleMaxClassificationLoss'):
        cfg.optimizer_cfg['lr'] = 0.01

    cfg.lr_scheduler = 'torch.optim.lr_scheduler.MultiStepLR'
    cfg.lr_scheduler_cfg = {'milestones': [int(0.4 * cfg.epochs_max), int(0.5 * cfg.epochs_max),
                                           int(0.6 * cfg.epochs_max), int(0.8 * cfg.epochs_max)],
                            'gamma': 0.2}

    return cfg

def sgdTruncation():
    cfg = TrainConfig

    cfg.optimizer = 'optimizers.SGDWithTruncation'
    cfg.optimizer_cfg = {'lr': 0.1, 'momentum': 0.9}

    if cfg.metrics[0][1] in ('WeightedRegressionLoss', 'SampleMaxClassificationLoss'):
        cfg.optimizer_cfg['lr'] = 0.01

    cfg.lr_scheduler = 'torch.optim.lr_scheduler.MultiStepLR'
    cfg.lr_scheduler_cfg = {'milestones': [int(0.4 * cfg.epochs_max), int(0.5 * cfg.epochs_max),
                                           int(0.6 * cfg.epochs_max), int(0.8 * cfg.epochs_max)],
                            'gamma': 0.2}

    return cfg

def wd(val='0.0001'):
    TrainConfig.optimizer_cfg['weight_decay'] = float(val)
    return TrainConfig


def seed(seed):
    cfg = TrainConfig
    cfg.seed = int(seed)
    return cfg


def averaging(av_type='all', burn_in_epochs=None):
    cfg = TrainConfig
    cfg.lr_scheduler = None
    cfg.lr_scheduler_cfg = {}
    cfg.batch_size = 125
    cfg.save_interval = 400
    if av_type == 'all':
        gamma = None
        if burn_in_epochs is None:
            burn_in_epochs = 10
    elif av_type == 'off':
        gamma = 1.0
        burn_in_epochs = 0
    else:
        gamma = float(av_type)
        if burn_in_epochs is None:
            burn_in_epochs = 0
    burn_in = (float(burn_in_epochs) * 50000) // cfg.batch_size
    cfg.optimizer = 'optimizers.SGDWithAveraging'
    cfg.optimizer_cfg = {'lr': 0.05, 'momentum': 0.9, 'weight_decay': 1e-4,
                         'gamma': gamma, 'burn_in': burn_in, 'module': None}
    cfg.epochs_max = 250
    return cfg


def anneal_lr(gamma=0.1, step=75):
    cfg = TrainConfig
    cfg.lr_scheduler = 'torch.optim.lr_scheduler.StepLR'
    cfg.lr_scheduler_cfg = {'step_size': int(step), 'gamma': float(gamma)}
    return cfg