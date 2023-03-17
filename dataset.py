import copy
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T


def get_dataloaders(name, train_halves):
    dataloaders = list()
    if name == 'cifar100':
        train_dset, test_dset = cifar100()
        if train_halves:
            train_dataset_part1, train_dataset_part2 = split_cifar100(train_dset)
            train_dataloder_part1 = train_dataloader(train_dataset_part1)
            train_dataloder_part2 = train_dataloader(train_dataset_part2)
            dataloaders.append(train_dataloder_part1)
            dataloaders.append(train_dataloder_part2)
        else:
            train_loader = train_dataloader(train_dset)
            dataloaders.append(train_loader)
        test_loader = test_dataloader(test_dset)
        dataloaders.append(test_loader)
    else:
        raise ValueError(f'invalida dataset name {name}')

    return dataloaders


def cifar100():
    CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR_STD = (0.2675, 0.2565, 0.2761)
    normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        normalize,
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])
    train_dset = torchvision.datasets.CIFAR100(root='/tmp', train=True,
                                               download=True, transform=train_transform)
    test_dset = torchvision.datasets.CIFAR100(root='/tmp', train=False,
                                              download=True, transform=test_transform)

    return train_dset, test_dset


def split_cifar100(train_dset):
    train_dataset_part1 = copy.deepcopy(train_dset)
    targets = torch.Tensor(train_dataset_part1.targets).to(torch.long)
    classes = torch.unique(targets)
    half = len(classes) // 2

    idx_part1 = targets < half
    train_dataset_part1.data = train_dataset_part1.data[idx_part1]
    train_dataset_part1.targets = targets[idx_part1]

    train_dataset_part2 = copy.deepcopy(train_dset)
    targets = torch.Tensor(train_dataset_part2.targets).to(torch.long)
    idx_part2 = targets >= half
    train_dataset_part2.data = train_dataset_part2.data[idx_part2]
    train_dataset_part2.targets = targets[idx_part2]

    return train_dataset_part1, train_dataset_part2


def train_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=True, num_workers=8)


def test_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False, num_workers=8)


if __name__ == '__main__':
    cifar100(get_parts=True)
