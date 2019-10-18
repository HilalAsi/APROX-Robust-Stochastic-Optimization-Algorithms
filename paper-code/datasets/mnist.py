import torchvision
import torchvision.transforms as transforms
import torch


def _randomize_labels(labels, seed):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    labels = list(torch.IntTensor(labels)[torch.randperm(len(labels))])
    torch.set_rng_state(rng_state)
    return labels


def MNIST(root='./data', train=True, download=True, augment=True,
            randomize_labels=False, randomize_label_seed=42, truncate=None):

    trans_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                      ]) if augment else trans_val
    dataset_train = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=trans_train)
    dataset_val = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=trans_val)


    if randomize_labels:
        dataset_train.train_labels = _randomize_labels(dataset_train.train_labels, randomize_label_seed)

    if truncate is not None:
        dataset_train.train_data = dataset_train.train_data[:truncate]
        dataset_train.train_labels = dataset_train.train_labels[:truncate]
        dataset_val.test_data = dataset_val.test_data[:truncate]
        dataset_val.test_labels = dataset_val.test_labels[:truncate]

    return dataset_train, dataset_val



