import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import dataset
import utils
import interpolate_networks
from train import evaluate, resnet18


def main():
    args = parse_args()
    utils.seed_everything(42)

    model0 = resnet18()
    utils.load_model(model0, args.weights1_path)
    model1 = resnet18()
    utils.load_model(model1, args.weights2_path)

    train_dataloader, test_dataloader = dataset.get_dataloaders(args.dataset, train_halves=False)
    if args.simulate_rehersal:
        train_dataset, _ = dataset.cifar100()
        random_idx = torch.randperm(len(train_dataset))[:500]
        train_dataset = torch.utils.data.Subset(train_dataset, random_idx)
        train_dataloader = dataset.train_dataloader(train_dataset)
    img_filename = 'interpolation'
    for arg in vars(args):
        img_filename += f'_{arg}={getattr(args, arg)}'
    interpolation_plot(model0, model1, train_dataloader, test_dataloader, img_filename)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=('cifar100', 'cifar10'), default='cifar100')
    parser.add_argument('--weights1_path', type=str, required=True)
    parser.add_argument('--weights2_path', type=str, required=True)
    parser.add_argument('--simulate_rehersal', action='store_true')

    args = parser.parse_args()
    return args


def interpolation_plot(model0, model1, train_loader, test_loader, img_filename):
    model0 = interpolate_networks.add_junctures(model0)
    model1 = interpolate_networks.add_junctures(model1)
    premuted_nework = resnet18()
    premuted_nework = interpolate_networks.add_junctures(premuted_nework)
    premuted_nework.load_state_dict(model1.state_dict())
    premuted_nework = interpolate_networks.permute_network(train_loader, test_loader, model0, premuted_nework)

    stats = {}
    alpha_grid = np.arange(0, 1.001, 0.02)
    # alpha_grid = np.arange(0, 1.001, 0.5)
    stats['vanilla'] = get_acc_barrier(alpha_grid, train_loader, test_loader, model0, model1)
    stats['permute'] = get_acc_barrier(alpha_grid, train_loader, test_loader, model0, premuted_nework)
    stats['renorm'] = get_acc_barrier(alpha_grid, train_loader, test_loader, model0, model1, reset_bn=True)
    stats['permute_renorm'] = get_acc_barrier(alpha_grid, train_loader, test_loader, model0, premuted_nework, reset_bn=True)

    for k in stats.keys():
        plt.plot(alpha_grid, stats[k], label=k)
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('test accuracy')

    plt.savefig('images/' + img_filename + '.png')
    plt.show()


def get_acc_barrier(alpha_grid, train_loader, test_loader, model1, model2, reset_bn=False):
    accs = []
    model = resnet18()
    model = interpolate_networks.add_junctures(model)

    for alpha in tqdm(alpha_grid):
        interpolate_networks.mix_weights(model, alpha, model1, model2)
        if reset_bn:
            interpolate_networks.reset_bn_stats(model, train_loader)
        model = interpolate_networks.remove_junctures(model)
        test_acc, _ = evaluate(model, test_loader)
        accs.append(test_acc)
    return accs


if __name__ == '__main__':
    main()
