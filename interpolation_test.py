# base on https://github.com/KellerJordan/REPAIR

import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os
import sys

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
import torchvision
import torchvision.transforms as T


def save_model(model, i):
    sd = model.state_dict()
    torch.save(model.state_dict(), '%s.pth.tar' % i)


def load_model(model, i):
    sd = torch.load('%s.pth.tar' % i)
    model.load_state_dict(sd)


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


train_dataset_part1 = torchvision.datasets.CIFAR100(root='/tmp', train=True,
                                                    download=True, transform=train_transform)
targets = torch.Tensor(train_dataset_part1.targets).to(torch.long)
idx_part1 = targets < 50
train_dataset_part1.data = train_dataset_part1.data[idx_part1]
train_dataset_part1.targets = targets[idx_part1]
train_dataloder_part1 = torch.utils.data.DataLoader(train_dataset_part1, batch_size=500, shuffle=True, num_workers=8)

train_dataset_part2 = torchvision.datasets.CIFAR100(root='/tmp', train=True,
                                                    download=True, transform=train_transform)
targets = torch.Tensor(train_dataset_part2.targets).to(torch.long)
idx_part2 = targets >= 50
train_dataset_part2.data = train_dataset_part2.data[idx_part2]
train_dataset_part2.targets = targets[idx_part2]
train_dataloder_part2 = torch.utils.data.DataLoader(train_dataset_part2, batch_size=500, shuffle=True, num_workers=8)

# evaluates accuracy


def evaluate(model, loader=test_loader):
    model.eval()
    correct = 0
    all = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            pred = outputs.argmax(dim=1)
            correct += (labels.cuda() == pred).sum().item()
            all += len(labels)
    return correct / all

# evaluates loss


def evaluate1(model, loader=test_loader):
    model.eval()
    losses = []
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            loss = F.cross_entropy(outputs, labels.cuda())
            losses.append(loss.item())
    return np.array(losses).mean()


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            #             self.shortcut = LambdaLayer(lambda x:
            #                                         F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, w=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = w*16

        self.conv1 = nn.Conv2d(3, w*16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w*16)
        self.layer1 = self._make_layer(block, w*16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, w*32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, w*64, num_blocks[2], stride=2)
        self.linear = nn.Linear(w*64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(w=1):
    return ResNet(BasicBlock, [3, 3, 3], w=w, num_classes=100)


def train(save_key, train_dataloader):
    model = resnet20(w=4).cuda()
    if save_key == 'resnet20x4_v2':  # continue training
        load_model(model, 'resnet20x4_v1')
    optimizer = SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=5e-4)
    # optimizer = Adam(model.parameters(), lr=0.05)

    # Adam seems to perform worse than SGD for training ResNets on CIFAR-10.
    # To make Adam work, we find that we need a very high learning rate: 0.05 (50x the default)
    # At this LR, Adam gives 1.0-1.5% worse accuracy than SGD.

    # It is not yet clear whether the increased interpolation barrier for Adam-trained networks
    # is simply due to the increased test loss of said networks relative to those trained with SGD.
    # We include the option of using Adam in this notebook to explore this question.

    EPOCHS = 100
    ne_iters = len(train_dataloader)
    lr_schedule = np.interp(np.arange(1+EPOCHS*ne_iters), [0, 5*ne_iters, EPOCHS*ne_iters], [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    losses = []
    for _ in tqdm(range(EPOCHS)):
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs.cuda())
                loss = loss_fn(outputs, labels.cuda())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
    print(f'model {save_key} accuracy = {evaluate(model)}')
    save_model(model, save_key)


# train('resnet20x4_v1', train_dataloder_part1)
train('resnet20x4_v2', train_dataloder_part2)

# given two networks net0, net1 which each output a feature map of shape NxCxWxH
# this will reshape both outputs to (N*W*H)xC
# and then compute a CxC correlation matrix between the outputs of the two networks


def run_corr_matrix(net0, net1, epochs=1, norm=True, loader=train_aug_loader):
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
    if norm:
        corr = cov / (torch.outer(std0, std1) + 1e-4)
        return corr
    else:
        return cov


def compute_perm_map(corr_mtx):
    # sort the (i, j) channel pairs by correlation
    nchan = corr_mtx.shape[0]
    triples = [(i, j, corr_mtx[i, j].item()) for i in range(nchan) for j in range(nchan)]
    triples = sorted(triples, key=lambda p: -p[2])
    # greedily find a matching
    perm_d = {}
    for i, j, c in triples:
        if not (i in perm_d.keys() or j in perm_d.values()):
            perm_d[i] = j
    perm_map = torch.tensor([perm_d[i] for i in range(nchan)])

    # qual_map will be a permutation of the indices in the order
    # of the quality / degree of correlation between the neurons found in the permutation.
    # this just for visualization purposes.
    qual_l = [corr_mtx[i, perm_map[i]].item() for i in range(nchan)]
    qual_map = torch.tensor(sorted(range(nchan), key=lambda i: -qual_l[i]))

    return perm_map, qual_map


def get_layer_perm1(corr_mtx, method='max_weight', vizz=False):
    if method == 'greedy':
        perm_map, qual_map = compute_perm_map(corr_mtx)
        if vizz:
            corr_mtx_viz = (corr_mtx[qual_map].T[perm_map[qual_map]]).T
            viz(corr_mtx_viz)
    elif method == 'max_weight':
        corr_mtx_a = corr_mtx.cpu().numpy()
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
        assert (row_ind == np.arange(len(corr_mtx_a))).all()
        perm_map = torch.tensor(col_ind).long()
    else:
        raise Exception('Unknown method: %s' % method)

    return perm_map

# returns the channel-permutation to make layer1's activations most closely
# match layer0's.


def get_layer_perm(net0, net1, method='max_weight', vizz=False):
    corr_mtx = run_corr_matrix(net0, net1)
    return get_layer_perm1(corr_mtx, method, vizz)

# modifies the weight matrices of a convolution and batchnorm
# layer given a permutation of the output channels


def permute_output(perm_map, conv, bn):
    pre_weights = [
        conv.weight,
        bn.weight,
        bn.bias,
        bn.running_mean,
        bn.running_var,
    ]
    for w in pre_weights:
        w.data = w[perm_map]

# modifies the weight matrix of a convolution layer for a given
# permutation of the input channels


def permute_input(perm_map, after_convs):
    if not isinstance(after_convs, list):
        after_convs = [after_convs]
    post_weights = [c.weight for c in after_convs]
    for w in post_weights:
        w.data = w[:, perm_map, :, :]


# Find neuron-permutation for each layer
model0 = resnet20(w=4).cuda()
model1 = resnet20(w=4).cuda()
load_model(model0, 'resnet20x4_v1')
load_model(model1, 'resnet20x4_v2')

evaluate(model0), evaluate(model1)


class Subnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        self = self.model
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return x


# corr = run_corr_matrix(Subnet(model0), Subnet(model1))
# perm_map1 = get_layer_perm1(corr)
perm_map = get_layer_perm(Subnet(model0), Subnet(model1))
permute_output(perm_map, model1.conv1, model1.bn1)
permute_output(perm_map, model1.layer1[0].conv2, model1.layer1[0].bn2)
permute_output(perm_map, model1.layer1[1].conv2, model1.layer1[1].bn2)
permute_output(perm_map, model1.layer1[2].conv2, model1.layer1[2].bn2)
permute_input(perm_map, [model1.layer1[0].conv1, model1.layer1[1].conv1, model1.layer1[2].conv1])
permute_input(perm_map, [model1.layer2[0].conv1, model1.layer2[0].shortcut[0]])


class Subnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        self = self.model
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        return x


perm_map = get_layer_perm(Subnet(model0), Subnet(model1))
permute_output(perm_map, model1.layer2[0].conv2, model1.layer2[0].bn2)
permute_output(perm_map, model1.layer2[0].shortcut[0], model1.layer2[0].shortcut[1])
permute_output(perm_map, model1.layer2[1].conv2, model1.layer2[1].bn2)
permute_output(perm_map, model1.layer2[2].conv2, model1.layer2[2].bn2)

permute_input(perm_map, [model1.layer2[1].conv1, model1.layer2[2].conv1])
permute_input(perm_map, [model1.layer3[0].conv1, model1.layer3[0].shortcut[0]])


class Subnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        self = self.model
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


perm_map = get_layer_perm(Subnet(model0), Subnet(model1))
permute_output(perm_map, model1.layer3[0].conv2, model1.layer3[0].bn2)
permute_output(perm_map, model1.layer3[0].shortcut[0], model1.layer3[0].shortcut[1])
permute_output(perm_map, model1.layer3[1].conv2, model1.layer3[1].bn2)
permute_output(perm_map, model1.layer3[2].conv2, model1.layer3[2].bn2)

permute_input(perm_map, [model1.layer3[1].conv1, model1.layer3[2].conv1])
model1.linear.weight.data = model1.linear.weight[:, perm_map]


class Subnet(nn.Module):
    def __init__(self, model, nb=9):
        super().__init__()
        self.model = model
        self.blocks = []
        self.blocks += list(model.layer1)
        self.blocks += list(model.layer2)
        self.blocks += list(model.layer3)
        self.blocks = nn.Sequential(*self.blocks)
        self.bn1 = model.bn1
        self.conv1 = model.conv1
        self.linear = model.linear
        self.nb = nb

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks[:self.nb](x)
        block = self.blocks[self.nb]
        x = block.conv1(x)
        x = block.bn1(x)
        x = F.relu(x)
        return x


blocks1 = []
blocks1 += list(model1.layer1)
blocks1 += list(model1.layer2)
blocks1 += list(model1.layer3)
blocks1 = nn.Sequential(*blocks1)

for nb in range(9):
    perm_map = get_layer_perm(Subnet(model0, nb=nb), Subnet(model1, nb=nb))
    block = blocks1[nb]
    permute_output(perm_map, block.conv1, block.bn1)
    permute_input(perm_map, [block.conv2])

save_model(model1, 'resnet20x4_v2_perm1')


def mix_weights(model, alpha, key0, key1):
    sd0 = torch.load('%s.pth.tar' % key0)
    sd1 = torch.load('%s.pth.tar' % key1)
    sd_alpha = {k: (1 - alpha) * sd0[k].cuda() + alpha * sd1[k].cuda()
                for k in sd0.keys()}
    model.load_state_dict(sd_alpha)

# use the train loader with data augmentation as this gives better results


def reset_bn_stats(model, epochs=1, loader=train_aug_loader):
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


model_a = resnet20(w=4).cuda(0)  # W_\alpha
mix_weights(model_a, 0.5, 'resnet20x4_v1', 'resnet20x4_v2_perm1')

print('Pre-reset:')
print('Accuracy=%.2f%%, Loss=%.3f' % (evaluate(model_a)/100, evaluate1(model_a)))

reset_bn_stats(model_a)
print('Post-reset:')
print('Accuracy=%.2f%%, Loss=%.3f' % (evaluate(model_a)/100, evaluate1(model_a)))
