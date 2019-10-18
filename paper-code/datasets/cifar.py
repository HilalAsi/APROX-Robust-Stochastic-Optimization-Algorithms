import torchvision
import torchvision.transforms as transforms
import torch


def _randomize_labels(labels, seed):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    labels = list(torch.IntTensor(labels)[torch.randperm(len(labels))])
    torch.set_rng_state(rng_state)
    return labels


def CIFAR10(root='./data', train=True, download=True, augment=True,
            randomize_labels=False, randomize_label_seed=42, truncate=None):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) if augment else transform_val

    train = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
    val = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform_val)

    if randomize_labels:
        train.train_labels = _randomize_labels(train.train_labels, randomize_label_seed)

    if truncate is not None:
        train.train_data = train.train_data[:truncate]
        train.train_labels = train.train_labels[:truncate]
        val.test_data = val.test_data[:truncate]
        val.test_labels = val.test_labels[:truncate]

    return train, val



def Big_CIFAR10(root='./data', train=True, download=True, augment=True,
            randomize_labels=False, randomize_label_seed=42, truncate=None):
    # returns images with size 224 x 224
    transform_val = transforms.Compose([
        #transforms.Scale(224),
        transforms.Scale(256),
        transforms.CenterCrop(224),
        #transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.Scale(224),
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #transforms.Scale(224)
    ]) if augment else transform_val

    train = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform_train)
    val = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform_val)

    if randomize_labels:
        train.train_labels = _randomize_labels(train.train_labels, randomize_label_seed)

    if truncate is not None:
        train.train_data = train.train_data[:truncate]
        train.train_labels = train.train_labels[:truncate]
        val.test_data = val.test_data[:truncate]
        val.test_labels = val.test_labels[:truncate]

    return train, val

def CIFAR100(root='./data', train=True, download=True, augment=True,
             randomize_labels=False, randomize_label_seed=42):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]) if augment else transform_val

    train = torchvision.datasets.CIFAR100(root=root, train=True, download=download, transform=transform_train)
    val = torchvision.datasets.CIFAR100(root=root, train=False, download=download, transform=transform_val)

    if randomize_labels:
        train.train_labels = _randomize_labels(train.train_labels, randomize_label_seed)

    return train, val
