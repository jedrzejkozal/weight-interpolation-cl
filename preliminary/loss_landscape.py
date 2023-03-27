from interpolate_networks import *
import torch
import torch.nn as nn
import torch.multiprocessing
import loss_landscape
import numpy as np
import matplotlib.pyplot as plt

import utils
import dataset
from train import resnet18


@torch.no_grad()
def main():
    utils.seed_everything(42)
    steps = 10

    loss_function = nn.CrossEntropyLoss()
    train_dataloader_part1, train_dataloader_part2, test_dataloader = dataset.get_dataloaders('cifar100', train_halves=True)

    model_part1 = 'resnet18_v3_50_epochs_part1'
    model_part2 = 'resnet18_v3_50_epochs_part2'

    model1 = resnet18()
    utils.load_model(model1, model_part1)
    model1.to('cuda')
    model2 = resnet18()
    utils.load_model(model2, model_part2)
    model2.to('cuda')
    model3 = resnet18()
    utils.load_model(model3, model_part1)
    model3.to('cuda')
    model3 = permute_nework(model2, model3, train_dataloader_part1, test_dataloader)
    model4 = resnet18()
    model4 = interpolation(model2, model3, model4, train_dataloader_part1)

    coordinates, dir_one, dir_two, start_point = loss_landscape.three_models(model1, model2, model3, model4, normalization='filter', steps=steps)
    coor1, coor2, coor3, coor4 = coordinates

    metric = loss_landscape.Loss(loss_function, train_dataloader_part1)
    # landscape = loss_landscape.random_plane(model, metric, normalization='filter', steps=steps)
    model = resnet18()
    load_params(model, start_point)
    model_wrapper = loss_landscape.wrap_model(model)
    start_point = model_wrapper.get_module_parameters()
    landscape = loss_landscape.get_grid(metric, steps, model_wrapper, start_point, dir_one, dir_two)

    # landscape = get_landscape()
    print(landscape)
    delta = 1.0
    border = steps // 2
    x = np.arange(-border, border, delta)
    y = np.arange(-border, border, delta)
    X, Y = np.meshgrid(x, y)

    loss_min = np.log10(landscape.min())
    loss_max = np.log10(landscape.max())
    levels = np.logspace(loss_min, loss_max, num=10)
    levels_finegrained = np.linspace(landscape.min(), min(2*landscape.min(), levels[1]), num=5)
    levels = [levels[0]] + list(levels_finegrained[1:-1]) + list(levels[1:])
    print(levels)
    print(levels_finegrained)
    # exit()

    CS = plt.contour(X, Y, landscape,)  # levels=levels)
    plt.clabel(CS, inline=True, fontsize=10, fmt='%1.1e')
    plt.plot(*coor1, 'rx', label='model 1')
    plt.plot(*coor2, 'bx', label='model 2')
    plt.plot(*coor3, 'gx', label='premuted model 1')
    plt.plot(*coor4, 'mo', label='interpolation')
    plt.legend()
    plt.show()


@torch.no_grad()
def main_single_model():
    utils.seed_everything(50)
    torch.multiprocessing.set_sharing_strategy('file_system')
    steps = 10

    model = resnet18()
    utils.load_model(model, 'resnet18_whole_50_epochs')
    model.to('cuda')

    loss_function = nn.CrossEntropyLoss(reduction='sum')
    train_dataloader, test_dataloader = dataset.get_dataloaders('cifar100', train_halves=False)

    metric = loss_landscape.Loss(loss_function, train_dataloader)
    landscape = loss_landscape.random_plane(model, metric, normalization='filter', steps=steps, distance=0.1)

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


def load_params(model, parameters):
    params_iter = iter(parameters)
    for name, _ in model.named_parameters():
        setattr(model, name, next(params_iter))


def permute_nework(source_network, premutation_nework, train_loader, test_loader):
    source_network = add_junctures(source_network)
    premutation_nework = add_junctures(premutation_nework)
    premutation_nework = permute_network(train_loader, test_loader, source_network, premutation_nework)
    premutation_nework = remove_junctures(premutation_nework)
    source_network = remove_junctures(source_network)
    return premutation_nework


def interpolation(sournce_network, premutation_nework, output_network, train_loader, alpha=0.5):
    mix_weights(output_network, alpha, sournce_network, premutation_nework)
    reset_bn_stats(output_network, train_loader)
    return output_network


def get_data(train_dataloader):
    X = []
    y = []
    for batch_X, batch_y in train_dataloader:
        X.append(batch_X)
        y.append(batch_y)
        # break
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    X = X.to('cuda')
    y = y.to('cuda')
    return X, y


if __name__ == '__main__':
    # main()
    main_single_model()
