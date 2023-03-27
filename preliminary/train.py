import torch.nn.functional as F
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchvision.models

# from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
from tqdm import tqdm

import dataset
import utils


def main():
    args = parse_args()
    utils.seed_everything(args.seed)

    dataloaders = dataset.get_dataloaders(args.dataset, args.train_halves)
    if args.train_halves:
        train_dataloader_part1, train_dataloader_part2, test_dataloader = dataloaders
        store_weights_path = args.store_weights_path + "_part1"
        train(train_dataloader_part1, test_dataloader,
              store_weights_path, args.load_weights_path)
        store_weights_path = args.store_weights_path + "_part2"
        train(train_dataloader_part2, test_dataloader,
              store_weights_path, args.load_weights_path)
    else:
        train_dataloader, test_dataloader = dataloaders
        train(train_dataloader, test_dataloader,
              args.store_weights_path, args.load_weights_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=('cifar100', 'cifar10'), default='cifar100')
    parser.add_argument('--train_halves', action='store_true')
    parser.add_argument('--load_weights_path', type=str, default=None)
    parser.add_argument('--store_weights_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


def train(train_dataloader, test_dataloader, store_weights_path, load_weights_path=None):
    model = resnet18()
    model.train()
    if load_weights_path:
        utils.load_model(model, load_weights_path)
    optimizer = SGD(model.parameters(), lr=0.008, momentum=0.8, weight_decay=1e-4)
    # optimizer = Adam(model.parameters(), lr=0.05)

    # Adam seems to perform worse than SGD for training ResNets on CIFAR-10.
    # To make Adam work, we find that we need a very high learning rate: 0.05 (50x the default)
    # At this LR, Adam gives 1.0-1.5% worse accuracy than SGD.

    # It is not yet clear whether the increased interpolation barrier for Adam-trained networks
    # is simply due to the increased test loss of said networks relative to those trained with SGD.
    # We include the option of using Adam in this notebook to explore this question.

    EPOCHS = 100
    # EPOCHS = 1
    # ne_iters = len(train_dataloader)
    # lr_schedule = np.interp(np.arange(1+EPOCHS*ne_iters), [0, 5*ne_iters, EPOCHS*ne_iters], [0, 1, 0])
    # print('lr_schedule = ', lr_schedule)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [80, 90, 95], 0.1)

    # scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    losses = []
    for _ in tqdm(range(EPOCHS)):
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            # with autocast():
            outputs = model(inputs.cuda())
            loss = loss_fn(outputs, labels.cuda())
            loss.backward()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

    train_acc, train_loss = evaluate(model, train_dataloader)
    test_acc, test_loss = evaluate(model, test_dataloader)
    print('model {} train loss = {:.4f} train accuracy = {:.4f} test loss = {:.4f} test accuracy = {:.4f}'.format(
        store_weights_path, train_loss, train_acc, test_loss, test_acc))
    utils.save_model(model, store_weights_path)


def resnet18():
    model = torchvision.models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model = model.cuda().eval()
    return model


def evaluate(model, loader):
    model.eval()
    losses = []
    correct = 0
    total = 0
    # with torch.no_grad(), autocast():
    for inputs, labels in loader:
        outputs = model(inputs.cuda())
        pred = outputs.argmax(dim=1)
        correct += (labels.cuda() == pred).sum().item()
        total += len(labels)
        loss = F.cross_entropy(outputs, labels.cuda())
        losses.append(loss.item())
    return correct / total, np.array(losses).mean()


if __name__ == '__main__':
    main()
