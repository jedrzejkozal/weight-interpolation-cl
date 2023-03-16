from torch.cuda.amp import autocast
from tqdm import tqdm
import scipy.optimize
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt

from train import resnet18, evaluate

import dataset
import utils


def main():
    args = parse_args()

    model0 = resnet18()
    utils.load_model(model0, 'resnet18_v1')
    model1 = resnet18()
    utils.load_model(model1, 'resnet18_v2')

    train_dataloader, test_dataloader = dataset.get_dataset(args.dataset, train_halves=False)
    interpolation(model0, model1, train_dataloader, test_dataloader)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=('cifar100', 'cifar10'), default='cifar100')
    # parser.add_argument('--weights1_path', type=str, required=True)
    # parser.add_argument('--weights2_path', type=str, required=True)

    args = parser.parse_args()
    return args


def interpolation(model0, model1, train_loader, test_loader):
    model_a = permute_network(train_loader, test_loader, model0, model1)

    stats = {}
    stats['vanilla'] = get_acc_barrier(train_loader, test_loader, 'resnet18j_v1', 'resnet18j_v2')
    stats['permute'] = get_acc_barrier(train_loader, test_loader, 'resnet18j_v1', 'resnet18j_v2_perm1')
    stats['renorm'] = get_acc_barrier(train_loader, test_loader, 'resnet18j_v1', 'resnet18j_v2', reset_bn=True)
    stats['permute_renorm'] = get_acc_barrier(train_loader, test_loader, 'resnet18j_v1', 'resnet18j_v2_perm1', reset_bn=True)

    for k in stats.keys():
        plt.plot(stats[k], label=k)
    plt.legend()
    plt.show()


def get_acc_barrier(train_loader, test_loader, model1_path, model2_path, reset_bn=False):
    accs = []
    alpha_grid = np.arange(0, 1.001, 0.02)
    # alpha_grid = np.arange(0, 1.001, 0.5)
    model = resnet18()
    model = add_junctures(model)

    for alpha in tqdm(alpha_grid):
        mix_weights(model, alpha, model1_path, model2_path)
        if reset_bn:
            reset_bn_stats(model, train_loader)
        test_acc, _ = evaluate(model, test_loader)
        accs.append(test_acc)
    return accs


def permute_network(train_aug_loader, test_loader, model0, model1):
    model0 = add_junctures(model0)
    model1 = add_junctures(model1)
    utils.save_model(model0, 'resnet18j_v1')
    utils.save_model(model1, 'resnet18j_v2')

    blocks0 = get_blocks(model0)
    blocks1 = get_blocks(model1)

    for k in range(1, len(blocks1)):
        block0 = blocks0[k]
        block1 = blocks1[k]
        subnet0 = nn.Sequential(blocks0[:k], block0.conv1, block0.bn1, block0.relu)
        subnet1 = nn.Sequential(blocks1[:k], block1.conv1, block1.bn1, block1.relu)
        perm_map = get_layer_perm(subnet0, subnet1, train_aug_loader)
        permute_output(perm_map, block1.conv1, block1.bn1)
        permute_input(perm_map, block1.conv2)

    last_kk = None
    perm_map = None

    for k in range(len(blocks1)):
        kk = get_permk(k)
        if kk != last_kk:
            perm_map = get_layer_perm(blocks0[:kk+1], blocks1[:kk+1], train_aug_loader)
            last_kk = kk

        if k > 0:
            permute_output(perm_map, blocks1[k].conv2, blocks1[k].bn2)
            shortcut = blocks1[k].downsample
            if isinstance(shortcut, nn.Conv2d):
                permute_output(perm_map, shortcut)
            else:
                permute_output(perm_map, shortcut[0], shortcut[1])
        else:
            permute_output(perm_map, model1.conv1, model1.bn1)

        if k+1 < len(blocks1):
            permute_input(perm_map, blocks1[k+1].conv1)
            shortcut = blocks1[k+1].downsample
            if isinstance(shortcut, nn.Conv2d):
                permute_input(perm_map, shortcut)
            else:
                permute_input(perm_map, shortcut[0])
        else:
            model1.fc.weight.data = model1.fc.weight[:, perm_map]

    test_acc = evaluate(model1, test_loader)[0]
    print('evaluate(model1) = ', test_acc)
    utils.save_model(model1, 'resnet18j_v2_perm1')

    model_a = add_junctures(resnet18())
    mix_weights(model_a, 0.5, 'resnet18j_v1', 'resnet18j_v2_perm1')
    reset_bn_stats(model_a, train_aug_loader)
    train_acc, train_loss = evaluate(model_a, train_aug_loader)
    test_acc, test_loss = evaluate(model_a, test_loader)
    print('train_acc = {}, train_loss = {}, test_acc = {}, test_loss = {}'.format(
        train_acc, train_loss, test_acc, test_loss))
    return model_a


def add_junctures(net):
    net1 = resnet18()
    net1.load_state_dict(net.state_dict())
    blocks = get_blocks(net1)[1:]
    for block in blocks:
        if block.downsample is not None:
            continue
        planes = len(block.bn2.weight)
        shortcut = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        shortcut.weight.data[:, :, 0, 0] = torch.eye(planes)
        block.downsample = shortcut
    return net1.cuda().eval()


def get_blocks(net):
    return nn.Sequential(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
                         *net.layer1, *net.layer2, *net.layer3, *net.layer4)


def get_layer_perm(net0, net1, train_dataloader):
    """
    returns the channel-permutation to make layer1's activations most closely
    match layer0's.
    """
    corr_mtx = run_corr_matrix(net0, net1, train_dataloader)
    return compute_permutation_matrix(corr_mtx)


def run_corr_matrix(net0, net1, loader, epochs=1):
    """
    given two networks net0, net1 which each output a feature map of shape NxCxWxH
    this will reshape both outputs to (N*W*H)xC
    and then compute a CxC correlation matrix between the outputs of the two networks
    """
    n = epochs*len(loader)
    mean0 = mean1 = std0 = std1 = None
    with torch.no_grad():
        net0.eval()
        net1.eval()
        for _ in range(epochs):
            for i, (images, _) in enumerate(tqdm(loader)):
                img_t = images.float().cuda()
                out0 = net0(img_t)
                out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                out0 = out0.reshape(-1, out0.shape[2]).double()

                out1 = net1(img_t)
                out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                out1 = out1.reshape(-1, out1.shape[2]).double()

                mean0_b = out0.mean(dim=0)
                mean1_b = out1.mean(dim=0)
                std0_b = out0.std(dim=0)
                std1_b = out1.std(dim=0)
                outer_b = (out0.T @ out1) / out0.shape[0]

                if i == 0:
                    mean0 = torch.zeros_like(mean0_b)
                    mean1 = torch.zeros_like(mean1_b)
                    std0 = torch.zeros_like(std0_b)
                    std1 = torch.zeros_like(std1_b)
                    outer = torch.zeros_like(outer_b)
                mean0 += mean0_b / n
                mean1 += mean1_b / n
                std0 += std0_b / n
                std1 += std1_b / n
                outer += outer_b / n

    cov = outer - torch.outer(mean0, mean1)
    corr = cov / (torch.outer(std0, std1) + 1e-4)
    return corr


def compute_permutation_matrix(corr_mtx):
    corr_mtx_a = corr_mtx.cpu().numpy()
    corr_mtx_a = np.nan_to_num(corr_mtx_a)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx_a))).all()
    perm_map = torch.tensor(col_ind).long()
    return perm_map


def get_permk(k):
    if k == 0:
        return 0
    elif k > 0 and k <= 2:
        return 2
    elif k > 2 and k <= 4:
        return 4
    elif k > 4 and k <= 6:
        return 6
    elif k > 6 and k <= 8:
        return 8
    else:
        raise Exception()


def permute_input(perm_map, conv):
    w = conv.weight
    w.data = w[:, perm_map, :, :]


def permute_output(perm_map, conv, bn=None):
    pre_weights = [conv.weight]
    if bn is not None:
        pre_weights.extend([bn.weight, bn.bias, bn.running_mean, bn.running_var])
    for w in pre_weights:
        w.data = w[perm_map]


def mix_weights(model, alpha, key0, key1):
    sd0 = torch.load('%s.pt' % key0)
    sd1 = torch.load('%s.pt' % key1)
    sd_alpha = {k: (1 - alpha) * sd0[k].cuda() + alpha * sd1[k].cuda()
                for k in sd0.keys()}
    model.load_state_dict(sd_alpha)


def reset_bn_stats(model, loader, epochs=1):
    """
    use the train loader with data augmentation as this gives better results
    """
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None  # use simple average
            m.reset_running_stats()
    # run a single train epoch with augmentations to recalc stats
    model.train()
    for _ in range(epochs):
        with torch.no_grad(), autocast():
            for images, _ in loader:
                output = model(images.cuda())


if __name__ == '__main__':
    main()
