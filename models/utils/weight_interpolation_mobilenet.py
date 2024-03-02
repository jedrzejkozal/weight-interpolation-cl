# code based on https://github.com/KellerJordan/REPAIR
import torch
import torch.nn as nn
import scipy.optimize
import numpy as np
from torchvision.models.mobilenetv2 import Conv2dNormActivation, InvertedResidual


def interpolate(sournce_network, premutation_nework, train_loader, device, alpha=0.5, permuation_epochs=1, batchnorm_epochs=1):
    premutation_nework = permute_network(train_loader, sournce_network, premutation_nework, device, epochs=permuation_epochs)
    mix_weights(premutation_nework, alpha, sournce_network, premutation_nework, device)
    reset_bn_stats(premutation_nework, train_loader, device, epochs=batchnorm_epochs)
    return premutation_nework


def permute_network(train_aug_loader, source_network, premuted_network, device, epochs=1):
    blocks0 = get_blocks(source_network)
    blocks1 = get_blocks(premuted_network)

    # first block has different number of layers in InvertedReusidual
    block0: InvertedResidual = blocks0[1]
    block1: InvertedResidual = blocks1[1]
    conv01, bn01 = block0.conv[0][0], block0.conv[0][1]
    conv11, bn11 = block1.conv[0][0], block1.conv[0][1]
    subnet0 = nn.Sequential(blocks0[:1], conv01, bn01, nn.ReLU6(inplace=True))
    subnet1 = nn.Sequential(blocks1[:1], conv11, bn11, nn.ReLU6(inplace=True))
    perm_map = get_layer_perm(subnet0, subnet1, train_aug_loader, device, epochs=epochs)
    permute_output(perm_map, conv11, bn11)
    conv12 = block1.conv[1]
    permute_input(perm_map, conv12)

    # permutaion between layers inside the blocks
    # iterate over inverted residual only
    for k in range(2, len(blocks1)-1):
        block0: InvertedResidual = blocks0[k]
        block1: InvertedResidual = blocks1[k]
        conv01, bn01 = block0.conv[0][0], block0.conv[0][1]
        conv11, bn11 = block1.conv[0][0], block1.conv[0][1]
        subnet0 = nn.Sequential(blocks0[:k], conv01, bn01, nn.ReLU6(inplace=True))
        subnet1 = nn.Sequential(blocks1[:k], conv11, bn11, nn.ReLU6(inplace=True))
        perm_map = get_layer_perm(subnet0, subnet1, train_aug_loader, device, epochs=epochs)
        permute_output(perm_map, conv11, bn11)
        conv12 = block1.conv[1][0]
        permute_input(perm_map, conv12)

        conv02, bn02 = block0.conv[1][0], block0.conv[1][1]
        conv12, bn12 = block1.conv[1][0], block1.conv[1][1]
        subnet0 = nn.Sequential(blocks0[:k], conv01, bn01, nn.ReLU6(inplace=True), conv02, bn02, nn.ReLU6(inplace=True))
        subnet1 = nn.Sequential(blocks1[:k], conv11, bn11, nn.ReLU6(inplace=True), conv12, bn12, nn.ReLU6(inplace=True))
        perm_map = get_layer_perm(subnet0, subnet1, train_aug_loader, device, epochs=epochs)
        permute_output(perm_map, conv12, bn12)
        conv13 = block1.conv[2]
        permute_input(perm_map, conv13)

    def get_fist(block):
        """return conv, batch_norm"""
        if type(block) == Conv2dNormActivation:
            return block[0], block[1]
        elif type(block) == InvertedResidual:
            return block.conv[0][0], block.conv[0][1]
        else:
            raise TypeError('Invalid type of block')

    def get_last(block):
        if type(block) == Conv2dNormActivation:
            return block[0], block[1]
        elif type(block) == InvertedResidual:
            return block.conv[-2], block.conv[-1]
        else:
            raise TypeError('Invalid type of block')

    last_kk = None
    perm_map = None

    # permutation between two neighboring blocks
    for k in range(len(blocks1)):
        kk = get_permk(k)
        if kk != last_kk:
            perm_map = get_layer_perm(blocks0[:kk+1], blocks1[:kk+1], train_aug_loader, device, epochs=epochs)
            last_kk = kk

        conv, bn = get_last(blocks1[k])
        permute_output(perm_map, conv, bn)

        if k+1 < len(blocks1):
            conv, _ = get_fist(blocks1[k+1])
            permute_input(perm_map, conv)
        else:
            premuted_network.classifier[1].weight.data = premuted_network.classifier[1].weight[:, perm_map]

    return premuted_network


def get_blocks(net):
    # return nn.Sequential(nn.Sequential(net.conv1, net.bn1),
    #                      *net.layer1, *net.layer2, *net.layer3, *net.layer4)
    return net.features


def get_layer_perm(net0, net1, train_dataloader, device, epochs=1):
    """
    returns the channel-permutation to make layer1's activations most closely
    match layer0's.
    """
    corr_mtx = run_corr_matrix(net0, net1, train_dataloader, device, epochs=epochs)
    return compute_permutation_matrix(corr_mtx)


def run_corr_matrix(net0, net1, loader, device, epochs=1):
    """
    given two networks net0, net1 which each output a feature map of shape NxCxWxH
    this will reshape both outputs to (N*W*H)xC
    and then compute a CxC correlation matrix between the outputs of the two networks
    """
    n = epochs * len(loader.dataset)
    mean0 = mean1 = std0 = std1 = None
    with torch.no_grad():
        net0.eval()
        net1.eval()
        for _ in range(epochs):
            for images, _ in loader:
                img_t = images.float().to(device)
                out0 = net0(img_t)
                out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                out0 = out0.reshape(-1, out0.shape[2]).double()

                out1 = net1(img_t)
                out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                out1 = out1.reshape(-1, out1.shape[2]).double()
                outer_b = out0.T @ out1

                if mean0 is None:
                    mean0 = torch.zeros(out0.shape[1:]).to(out0.device)
                    mean1 = torch.zeros(out1.shape[1:]).to(out1.device)
                    outer = torch.zeros_like(outer_b)
                mean0 += out0.sum(dim=0)
                mean1 += out1.sum(dim=0)

                outer += outer_b

        mean0 = mean0 / n
        mean1 = mean1 / n
        outer = outer / n

        for _ in range(epochs):
            for images, _ in loader:
                img_t = images.float().to(device)
                out0 = net0(img_t)
                out0 = out0.reshape(out0.shape[0], out0.shape[1], -1).permute(0, 2, 1)
                out0 = out0.reshape(-1, out0.shape[2]).double()

                out1 = net1(img_t)
                out1 = out1.reshape(out1.shape[0], out1.shape[1], -1).permute(0, 2, 1)
                out1 = out1.reshape(-1, out1.shape[2]).double()

                if std0 is None:
                    std0 = torch.zeros(out0.shape[1:]).to(out0.device)
                    std1 = torch.zeros(out1.shape[1:]).to(out1.device)

                std0 += ((out0 - mean0)**2).sum(dim=0)
                std1 += ((out1 - mean1)**2).sum(dim=0)

        std0 = torch.sqrt(std0 / (n-1))
        std1 = torch.sqrt(std1 / (n-1))

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
    if k == 1:
        return 1
    elif k > 1 and k <= 3:
        return 3
    elif k > 3 and k <= 6:
        return 6
    elif k > 6 and k <= 10:
        return 10
    elif k > 10 and k <= 13:
        return 13
    elif k > 13 and k <= 16:
        return 16
    elif k > 16 and k <= 17:
        return 17
    elif k > 17 and k <= 18:
        return 18
    else:
        raise Exception()


def permute_input(perm_map, conv):
    if conv.groups > 1:
        return
    w = conv.weight
    w.data = w[:, perm_map, :, :]


def permute_output(perm_map, conv, bn=None):
    pre_weights = [conv.weight]
    if bn is not None:
        pre_weights.extend([bn.weight, bn.bias, bn.running_mean, bn.running_var])
    for w in pre_weights:
        w.data = w[perm_map]


def mix_weights(model, alpha, model0, model1, device):
    state_dict0 = model0.state_dict()
    state_dict1 = model1.state_dict()
    sd_alpha = {k: (1 - alpha) * state_dict0[k].to(device) + alpha * state_dict1[k].to(device)
                for k in state_dict0.keys()}
    model.load_state_dict(sd_alpha)


def reset_bn_stats(model, loader, device, epochs=1):
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
        with torch.no_grad():
            for images, _ in loader:
                output = model(images.to(device))
