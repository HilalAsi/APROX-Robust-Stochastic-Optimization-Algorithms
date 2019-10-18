import torch
from os.path import join
import numpy as np



class TrainConfig(object):
    results_dir = 'results'
    checkpoint_format = 'net_step_%07d.t7'  # if changing this, three first characters must be constant!!!

    pre_trained = False #specify if want to use a pre_trained model

    batch_size = 10
    batch_size_val = None

    # save_interval = 100  # steps
    epochs_max = 100 #100

    gpus = list(range(torch.cuda.device_count()))
    loader_workers = 2
    ephemeral_params = ('results_dir', 'gpus', 'loader_workers', 'ephemeral_params')

    model = 'GoogLeNet'
    model_cfg = {}

    dataset = 'CIFAR10'
    dataset_cfg = {}

    optimizer = 'torch.optim.SGD'
    optimizer_cfg = {}

    # metrics is a list of 3-tuples, each of the form (name, class, config), where name is a string with the name of
    # the metric to be displayed (i.e. 'accuracy'), class is a string with the name of the class implementing the
    # metric (i.e. 'TopKAccuracy'), and config is a dictionary of arguments to the class constructor. "dataset" is
    # a special key which the training script replaces with the training set at runtime
    # note: the first metric in the list is the loss we use for training (which must therefore be differentiable)
    metrics = [('loss', 'nn.CrossEntropyLoss', {}),
               ('accuracy', 'TopKAccuracy', {'k': 1})]

    lr_scheduler = 'torch.optim.lr_scheduler.StepLR'
    lr_scheduler_cfg = {'step_size': 20, 'gamma': 0.2}  # decays the learning rate by gamma every step_size
    seed = None
    use_cuda = True
    # config name is automatically generated!
    name = ''


def save_config(cfg=TrainConfig, save_dir=None):
    if save_dir is None:
        save_dir = join(cfg.results_dir, cfg.name)
    to_save = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
    torch.save(to_save, join(save_dir, 'config.t7'))
    # TODO: if saved config exists, make sure there is no option clash


def load_config(load_dir, cfg=TrainConfig):
    loaded = torch.load(join(load_dir, 'config.t7'))
    for k, v in loaded.items():
        setattr(cfg, k, v)
    return cfg
