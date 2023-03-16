import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as T


def get_dataset(name, train_halves):
    if name == 'cifar100':
        dataloaders = cifar100(train_halves)
    else:
        raise ValueError(f'invalida dataset name {name}')

    return dataloaders


def cifar100(train_halves=False):
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

    train_aug_loader = torch.utils.data.DataLoader(train_dset, batch_size=500, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=500, shuffle=False, num_workers=8)

    if not train_halves:
        return train_aug_loader, test_loader

    train_dataset_part1 = torchvision.datasets.CIFAR100(root='/tmp', train=True,
                                                        download=True, transform=train_transform)
    targets = torch.Tensor(train_dataset_part1.targets).to(torch.long)
    classes = torch.unique(targets)
    half = len(classes) // 2

    idx_part1 = targets < half
    train_dataset_part1.data = train_dataset_part1.data[idx_part1]
    train_dataset_part1.targets = targets[idx_part1]
    train_dataloder_part1 = torch.utils.data.DataLoader(train_dataset_part1, batch_size=500, shuffle=True, num_workers=8)

    train_dataset_part2 = torchvision.datasets.CIFAR100(root='/tmp', train=True,
                                                        download=True, transform=train_transform)
    targets = torch.Tensor(train_dataset_part2.targets).to(torch.long)
    idx_part2 = targets >= half
    train_dataset_part2.data = train_dataset_part2.data[idx_part2]
    train_dataset_part2.targets = targets[idx_part2]
    train_dataloder_part2 = torch.utils.data.DataLoader(train_dataset_part2, batch_size=500, shuffle=True, num_workers=8)

    return (train_dataloder_part1, train_dataloder_part2), test_loader


if __name__ == '__main__':
    cifar100(get_parts=True)
