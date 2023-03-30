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
    dataset = get_dataset(args)

    num_tasks = 20
    results = []
    result_labels = []

    device = 'cuda'
    buffer = Buffer(args.buffer_size, device)

    artifact_path = '/home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/0/743680f7bbf94924abc7629aa6f724ac/artifacts/{}_task_{}/{}'
    # alpha_grid = np.arange(0, 1.001, 0.1)
    alpha_grid = np.arange(0, 1.001, 0.02)
    # alpha_grid = np.arange(0, 1.001, 0.5)
    for t in tqdm(range(num_tasks)):
        train_loader, test_loader = dataset.get_data_loaders()

        if t in (1, 4, 9, 14, 19):
            buffer_dataloder = get_buffer_dataloder(buffer, dataset)
            interpolation_accs = []

            for alpha in alpha_grid:
                net_t = torch.load(artifact_path.format('net_model', t, 'net.pt'))
                old_t = torch.load(artifact_path.format('old_model', t, 'old_model.pt'))
                new_model = interpolate(net_t, old_t, buffer_dataloder, alpha=alpha)
                acc, _ = evaluate(new_model, dataset)
                interpolation_accs.append(acc)

            results.append(interpolation_accs)
            result_labels.append(t)

        for _, labels, not_aug_inputs in train_loader:
            real_batch_size = not_aug_inputs.shape[0]
            buffer.add_data(examples=not_aug_inputs,
                            labels=labels[:real_batch_size])

    for t, interpolation_accs in zip(result_labels, results):
        plt.plot(alpha_grid, interpolation_accs, label=t)
    plt.legend()
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.show()


def get_buffer_dataloder(buffer, dataset):
    buf_inputs, buf_labels = buffer.get_data(len(buffer), transform=dataset.get_transform())
    buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
    buffer_dataloder = torch.utils.data.DataLoader(buffer_dataset, batch_size=500, num_workers=0)
    return buffer_dataloder


def evaluate(model, dataset):
    model.cuda()
    model.eval()
    losses = []
    correct = 0
    total = 0

    # test_dataloders = [dataset.test_loaders[-1]]
    # test_dataloders = dataset.test_loaders[:-1]
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
    return correct / total, np.array(losses).mean()


if __name__ == '__main__':
    main()
