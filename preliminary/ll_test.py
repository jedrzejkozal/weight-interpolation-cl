import torch
import torch.nn as nn
import numpy as np
import loss_landscapes
import loss_landscapes.metrics
import matplotlib.pyplot as plt

import dataset
import utils

from train import resnet18


def main():
    utils.seed_everything(42)
    torch.multiprocessing.set_sharing_strategy('file_system')
    steps = 10

    model = resnet18()
    model.cpu()
    utils.load_model(model, 'resnet18_whole_50_epochs')

    loss_function = nn.CrossEntropyLoss()
    train_dataloader, test_dataloader = dataset.get_dataloaders('cifar100', train_halves=False)
    X_train, y_train = get_data(train_dataloader)

    metric = loss_landscapes.metrics.Loss(loss_function, X_train, y_train)
    landscape = loss_landscapes.random_plane(model, metric, normalization='filter', steps=steps, distance=0.1)

    print(landscape)
    delta = 1.0
    border = steps // 2
    x = np.arange(-border, border, delta)
    y = np.arange(-border, border, delta)
    X, Y = np.meshgrid(x, y)
    loss_min = np.log10(landscape.min())
    loss_max = np.log10(landscape.max())
    levels = np.logspace(loss_min, loss_max, num=10)
    levels_finegrained = np.linspace(landscape.min(), 2*landscape.min(), num=5)
    levels = [levels[0]] + list(levels_finegrained[1:]) + list(levels[1:])
    print(levels)
    print(levels_finegrained)

    plt.contour(X, Y, landscape, levels=levels)
    plt.show()


def get_data(train_dataloader):
    X = []
    y = []
    for batch_X, batch_y in train_dataloader:
        X.append(batch_X)
        y.append(batch_y)
        # break
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    # X = X.to('cuda')
    # y = y.to('cuda')
    return X, y


if __name__ == '__main__':
    main()
