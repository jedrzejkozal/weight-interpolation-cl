import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import copy

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models.utils.weight_interpolation import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning with Weight Interpolation')
    parser.add_argument('--interpolation_alpha', type=float, default=0.5,
                        help='interpolation alpha')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Clewi(ContinualModel):
    NAME = 'clewi'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.old_model = self.deepcopy_model(backbone)
        self.interpolation_alpha = args.interpolation_alpha
        self.first_task = True

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        # if self.first_task:
        #     self.first_task = False
        #     self.old_model = self.deepcopy_model(self.net)
        #     return
        buffer_dataloder = self.get_buffer_dataloder()
        # torch.save(self.old_model, 'old_model.pt')
        # torch.save(self.net, 'net.pt')

        # self.interpolation_plot(dataset, buffer_dataloder)

        self.old_model = interpolate(self.net, self.old_model, buffer_dataloder, alpha=self.interpolation_alpha)
        # self.train_model_after_interpolation(buffer_dataloder)
        self.net = self.deepcopy_model(self.old_model)
        self.opt = self.opt.__class__(self.net.parameters(), **self.opt.defaults)
        self.opt.zero_grad()

    def interpolation_plot(self, dataset, buffer_dataloder):
        alpha_grid = np.arange(0, 1.001, 0.02)
        net = self.deepcopy_model(self.net)
        old = self.deepcopy_model(self.old_model)
        interpolation_accs = []
        for alpha in alpha_grid:
            new_model = interpolate(net, old, buffer_dataloder, alpha=alpha)
            acc = evaluate(new_model, dataset, self.device)
            interpolation_accs.append(acc)
        print(interpolation_accs)

    def get_buffer_dataloder(self):
        buf_inputs, buf_labels = self.buffer.get_data(len(self.buffer), transform=self.transform)
        buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        buffer_dataloder = torch.utils.data.DataLoader(buffer_dataset, batch_size=32, num_workers=0)
        return buffer_dataloder

    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        model_copy.load_state_dict(model.state_dict())
        return model_copy

    def train_model_after_interpolation(self, datalodaer):
        self.old_model.train()
        self.old_model = self.freeze_weights(self.old_model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.old_model.parameters(), lr=0.0001)

        for input, target in datalodaer:
            optimizer.zero_grad()
            input = input.to(self.device)
            target = target.to(self.device)
            y_pred = self.old_model(input)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()

        self.old_model = self.unfreeze_weights(self.old_model)

    def freeze_weights(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not ('linear' in name or 'classifier' in name):
                param.requires_grad = False
        return model

    def unfreeze_weights(self, model: nn.Module):
        for _, param in model.named_parameters():
            param.requires_grad = True
        return model


def evaluate(network: ContinualModel, dataset, device, last=False):
    status = network.training
    network.eval()
    network.cuda()
    accs = []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, total = 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

        accs.append(correct / total * 100)

    network.train(status)
    return accs
