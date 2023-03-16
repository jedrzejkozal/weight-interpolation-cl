import os

from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.optimize

import torch
import torch.utils.data
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models


def save_model(model, i):
    sd = model.state_dict()
    torch.save(model.state_dict(), '%s.pt' % i)


def load_model(model, i):
    sd = torch.load('%s.pt' % i)
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


def evaluate(model, loader=test_loader, tta=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            if tta:
                outputs = (outputs + model(inputs.flip(3)))/2
            pred = outputs.argmax(dim=1)
            correct += (labels.cuda() == pred).sum().item()
            total += len(labels)
    return correct / total

# evaluates acc and loss


def evaluate2(model, loader=test_loader, tta=False):
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            if tta:
                outputs = (outputs + model(inputs.flip(3)))/2
            pred = outputs.argmax(dim=1)
            correct += (labels.cuda() == pred).sum().item()
            total += len(labels)
            loss = F.cross_entropy(outputs, labels.cuda())
            losses.append(loss.item())
    return correct / total, np.array(losses).mean()


def full_eval(model):
    tr_acc, tr_loss = evaluate2(model, loader=train_aug_loader)
    te_acc, te_loss = evaluate2(model, loader=test_loader)
    return (100*tr_acc, tr_loss, 100*te_acc, te_loss)


def resnet18():
    model = torchvision.models.resnet18(pretrained=False)
    return model.cuda().eval()


def get_blocks(net):
    return nn.Sequential(nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
                         *net.layer1, *net.layer2, *net.layer3, *net.layer4)


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


# Matching code

# given two networks net0, net1 which each output a feature map of shape NxCxWxH
# this will reshape both outputs to (N*W*H)xC
# and then compute a CxC correlation matrix between the outputs of the two networks
def run_corr_matrix(net0, net1, epochs=1, loader=train_aug_loader):
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


def get_layer_perm1(corr_mtx):
    corr_mtx_a = corr_mtx.cpu().numpy()
    corr_mtx_a = np.nan_to_num(corr_mtx_a)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_mtx_a, maximize=True)
    assert (row_ind == np.arange(len(corr_mtx_a))).all()
    perm_map = torch.tensor(col_ind).long()
    return perm_map

# returns the channel-permutation to make layer1's activations most closely
# match layer0's.


def get_layer_perm(net0, net1):
    corr_mtx = run_corr_matrix(net0, net1)
    return get_layer_perm1(corr_mtx)


def permute_output(perm_map, conv, bn=None):
    pre_weights = [conv.weight]
    if bn is not None:
        pre_weights.extend([bn.weight, bn.bias, bn.running_mean, bn.running_var])
    for w in pre_weights:
        w.data = w[perm_map]


def permute_input(perm_map, conv):
    w = conv.weight
    w.data = w[:, perm_map, :, :]


def train(save_key, train_dataloader):
    model = resnet18()
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


# train('resnet18_v1', train_aug_loader)  # train_dataloder_part1
# train('resnet18_v2', train_aug_loader)  # train_dataloder_part2

# sd = torch.load('/persist/kjordan/notebooks/permutations/imagenet/0045eef3.pt')
# model.load_state_dict(sd)
# save_model(model, 'imagenet/resnet18_v1')

# sd = torch.load('/persist/kjordan/notebooks/permutations/imagenet/00ab6ff6.pt')
# model.load_state_dict(sd)
# save_model(model, 'imagenet/resnet18_v2')

model0 = resnet18()
model1 = resnet18()
load_model(model0, 'resnet18_v1')
load_model(model1, 'resnet18_v2')

model0 = add_junctures(model0)
model1 = add_junctures(model1)
save_model(model0, 'resnet18j_v1')
save_model(model1, 'resnet18j_v2')

blocks0 = get_blocks(model0)
blocks1 = get_blocks(model1)
evaluate(model0), evaluate(model1)

for k in range(1, len(blocks1)):
    block0 = blocks0[k]
    block1 = blocks1[k]
    subnet0 = nn.Sequential(blocks0[:k], block0.conv1, block0.bn1, block0.relu)
    subnet1 = nn.Sequential(blocks1[:k], block1.conv1, block1.bn1, block1.relu)
    perm_map = get_layer_perm(subnet0, subnet1)
    permute_output(perm_map, block1.conv1, block1.bn1)
    permute_input(perm_map, block1.conv2)


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


last_kk = None
perm_map = None

for k in range(len(blocks1)):
    kk = get_permk(k)
    if kk != last_kk:
        perm_map = get_layer_perm(blocks0[:kk+1], blocks1[:kk+1])
        last_kk = kk
#     perm_map = get_layer_perm(blocks0[:k+1], blocks1[:k+1])

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


print('evaluate(model1) = ', evaluate(model1))
save_model(model1, 'resnet18j_v2_perm1')


def mix_weights(model, alpha, key0, key1):
    sd0 = torch.load('%s.pt' % key0)
    sd1 = torch.load('%s.pt' % key1)
    sd_alpha = {k: (1 - alpha) * sd0[k].cuda() + alpha * sd1[k].cuda()
                for k in sd0.keys()}
    model.load_state_dict(sd_alpha)

# use the train loader with data augmentation as this gives better results


def reset_bn_stats(model, epochs=1, loader=train_aug_loader):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None
            m.reset_running_stats()
    # run a single train epoch with augmentations to recalc stats
    model.train()
    for _ in range(epochs):
        with torch.no_grad(), autocast():
            for images, _ in loader:
                images = images.cuda()
                output = model(images)


model_a = add_junctures(resnet18())
mix_weights(model_a, 0.5, 'resnet18j_v1', 'resnet18j_v2_perm1')
reset_bn_stats(model_a)
full_eval(model_a)


xx = np.arange(0, 1.001, 0.02)

stats = {}

bb = []
for alpha in tqdm(xx):
    mix_weights(model_a, alpha, 'resnet18j_v1', 'resnet18j_v2')
    bb.append(full_eval(model_a))
stats['vanilla'] = bb

bb = []
for alpha in tqdm(xx):
    mix_weights(model_a, alpha, 'resnet18j_v1', 'resnet18j_v2_perm1')
    bb.append(full_eval(model_a))
stats['permute'] = bb

bb = []
for alpha in tqdm(xx):
    mix_weights(model_a, alpha, 'resnet18j_v1', 'resnet18j_v2')
    reset_bn_stats(model_a)
    bb.append(full_eval(model_a))
stats['renorm'] = bb

bb = []
for alpha in tqdm(xx):
    mix_weights(model_a, alpha, 'resnet18j_v1', 'resnet18j_v2_perm1')
    reset_bn_stats(model_a)
    bb.append(full_eval(model_a))
stats['permute_renorm'] = bb

p = 'resnet18j_imagenet_barrier50.pt'
torch.save(stats, p)


for k in stats.keys():
    cc = [b[2] for b in stats[k]]
    plt.plot(cc, label=k)
plt.legend()
plt.show()
