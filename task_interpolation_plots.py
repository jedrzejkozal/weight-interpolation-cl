from datasets.seq_cifar100 import TCIFAR100, base_path
from torchvision.transforms import transforms
import torch.utils.data
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from main import parse_args
from models.utils.weight_interpolation import interpolate
from datasets import ContinualDataset, get_dataset
from utils.buffer import Buffer


def main():
    args = parse_args()

    run_path = '/home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/0/336b81ee3fca45d1a12eec09736dae39'
    artifact_path = run_path + '/artifacts/{}_task_{}/{}'
    seed_file_path = run_path + '/params/seed'
    with open(seed_file_path, 'r') as f:
        seed = f.read()
        seed = int(seed)
    args.seed = seed

    plot(args, artifact_path, 1)
    plot(args, artifact_path, 2, evaluate_buffer=True)
    plot(args, artifact_path, 3, evaluate_previous=True)
    plot(args, artifact_path, 4, evaluate_last=True)

    plt.show()


def plot(args, artifact_path, suplot_idx, evaluate_last=False, evaluate_previous=False, evaluate_buffer=False):
    num_tasks = 20
    results_acc = []
    results_loss = []
    result_labels = []

    dataset = get_dataset(args)
    device = 'cuda'
    buffer = Buffer(args.buffer_size, device)

    # alpha_grid = np.arange(0, 1.001, 0.1)
    alpha_grid = np.arange(0, 1.001, 0.02)
    # alpha_grid = np.arange(0, 1.001, 0.5)
    for t in tqdm(range(num_tasks)):
        train_loader, test_loader = dataset.get_data_loaders()

        if t in (1, 2, 3, 4):
            buffer_dataloder = get_buffer_dataloder(buffer, dataset)
            interpolation_accs = []
            interpolation_losses = []

            for alpha in alpha_grid:
                net_t = torch.load(artifact_path.format('net_model', t, 'net.pt'))
                old_t = torch.load(artifact_path.format('old_model', t, 'old_model.pt'))
                new_model = interpolate(net_t, old_t, buffer_dataloder, args.device, alpha=alpha)
                acc, loss = evaluate(new_model, dataset, buffer_dataloder, evaluate_last, evaluate_previous, evaluate_buffer)
                interpolation_accs.append(acc)
                interpolation_losses.append(loss)

            results_acc.append(interpolation_accs)
            results_loss.append(interpolation_losses)
            result_labels.append(t)

        for _, labels, not_aug_inputs in train_loader:
            real_batch_size = not_aug_inputs.shape[0]
            buffer.add_data(examples=not_aug_inputs,
                            labels=labels[:real_batch_size])

    plt.figure(1)
    plt.subplot(2, 2, suplot_idx)
    for t, interpolation_accs in zip(result_labels, results_acc):
        plt.plot(alpha_grid, interpolation_accs, label=t)
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('accuracy')

    plt.figure(2)
    plt.subplot(2, 2, suplot_idx)
    for t, interpolation_accs in zip(result_labels, results_loss):
        plt.plot(alpha_grid, interpolation_accs, label=t)
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('loss')


def get_buffer_dataloder(buffer, dataset):
    buf_inputs, buf_labels = buffer.get_data(len(buffer), transform=dataset.get_transform())
    buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
    buffer_dataloder = torch.utils.data.DataLoader(buffer_dataset, batch_size=500, num_workers=0)
    return buffer_dataloder


def evaluate(model, dataset, buffer_dataloder, evaluate_last=False, evaluate_previous=False, evaluate_buffer=False):
    model.cuda()
    model.eval()
    losses = []
    correct = 0
    total = 0

    if evaluate_buffer:
        test_dataloders = [buffer_dataloder]
    elif evaluate_last:
        test_dataloders = [dataset.test_loaders[-1]]
    elif evaluate_previous:
        test_dataloders = dataset.test_loaders[:-1]
    else:
        test_dataloders = dataset.test_loaders

    for test_dataloder in test_dataloders:
        for inputs, labels in test_dataloder:
            with torch.no_grad():
                outputs = model(inputs.cuda())
                pred = outputs.argmax(dim=1)
                correct += (labels.cuda() == pred).sum().item()
                total += len(labels)
                loss = F.cross_entropy(outputs, labels.cuda(), reduction='sum')
                losses.append(loss.item())
                # break
    return correct / total, sum(losses) / total


if __name__ == '__main__':
    main()
