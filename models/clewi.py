import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import copy

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models.utils.weight_interpolation import *
from models.utils.hessian_trace import hessian_trace


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning with Weight Interpolation')
    parser.add_argument('--interpolation_alpha', type=float, default=0.5,
                        help='interpolation alpha')
    parser.add_argument('--permuation_epochs', type=int, default=1)
    parser.add_argument('--batchnorm_epochs', type=int, default=1)
    parser.add_argument('--debug_interpolation', action='store_true')
    parser.add_argument('--lambda_grad_penalty', type=float, default=0.1)

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
        self.old_model = None
        self.interpolation_alpha = args.interpolation_alpha

        self.first_task = True
        self.t = 0

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        inputs.requires_grad = True
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        penalty = self.args.lambda_grad_penalty * self.calc_gradient_penalty(inputs, outputs)
        if not self.first_task and not torch.isnan(penalty):
            loss += penalty
        if torch.isnan(penalty):
            print('isnan')

        # ht = hessian_trace(self.net, loss, self.device, 10)
        # print(ht)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    @staticmethod
    def calc_gradient_penalty(x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)
        grad_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((grad_norm - 1) ** 2).mean()
    #     gradient_penalty = F.relu(grad_norm - 1).mean()

        return gradient_penalty

    def end_task(self, dataset):
        print('\n\n')
        # if self.t == 1:
        #     self.interpolation_alpha = 0.2
        # elif self.t == 2:
        #     self.interpolation_alpha = 0.25
        # elif self.t == 3:
        #     self.interpolation_alpha = 0.3
        # elif self.t == 5:
        #     self.interpolation_alpha = 0.33
        # elif self.t == 6:
        #     self.interpolation_alpha = 0.35
        # elif self.t == 9:
        #     self.interpolation_alpha = 0.37
        # self.t += 1

        # torch.cuda.empty_cache()
        # buffer_dataloder = self.get_buffer_dataloder(batch_size=256)
        # for input, target in buffer_dataloder:
        #     input = input.to(self.device)
        #     target = target.to(self.device)
        #     y_pred = self.net(input)
        #     loss = self.loss(y_pred, target)
        #     ht = hessian_trace(self.net, loss, self.device, 10)
        #     print('new model hessian trace = ', ht)

        # if self.old_model is not None:
        #     torch.cuda.empty_cache()
        #     for input, target in buffer_dataloder:
        #         input = input.to(self.device)
        #         target = target.to(self.device)
        #         y_pred = self.old_model(input)
        #         loss = self.loss(y_pred, target)
        #         ht = hessian_trace(self.old_model, loss, self.device, 10)
        #         print('old model hessian trace = ', ht)

        print('end_task call')
        print('interpolation_alpha = ', self.interpolation_alpha)

        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
            return

        torch.save(self.old_model, 'old_model.pt')
        torch.save(self.net, 'net.pt')

        L2_dist, diff_average, diff_sum, max_diff, min_diff, n = self.weight_difference(self.net, self.old_model)
        print(f'weight_difference: L2 = {L2_dist}, diff_average = {diff_average}, diff_sum = {diff_sum}, max_diff = {max_diff}, min_diff = {min_diff}, n = {n}')
        print('losses before interpolation:')
        _, losses, ns = evaluate(self.old_model, dataset, self.device, reduction='sum')
        print(f'old model: losses = {losses}, ns = {ns}')
        _, losses, ns = evaluate(self.net, dataset, self.device, reduction='sum')
        print(f'new model: losses = {losses}, ns = {ns}')

        buffer_dataloder = self.get_buffer_dataloder()
        if self.args.debug_interpolation:
            self.interpolation_plot(dataset, buffer_dataloder)
        self.old_model = interpolate(self.net, self.old_model, buffer_dataloder, self.device, alpha=self.interpolation_alpha,
                                     permuation_epochs=self.args.permuation_epochs, batchnorm_epochs=self.args.batchnorm_epochs)
        self.net = self.deepcopy_model(self.old_model)
        print('losses after interpolation:')
        _, losses, ns = evaluate(self.old_model, dataset, self.device, reduction='sum')
        print(f'interpolated model: losses = {losses}, ns = {ns}')
        # self.train_model_after_interpolation(buffer_dataloder)
        self.opt = self.opt.__class__(self.net.parameters(), **self.opt.defaults)
        self.opt.zero_grad()

    def weight_difference(self, net1, net2):
        diff_sum = 0.0
        diff_sq_sum = 0.0
        max_diff = 0.0
        min_diff = 100000000.0
        n = 0
        with torch.no_grad():
            for p1, p2 in zip(net1.state_dict().values(), net2.state_dict().values()):
                diff = p1 - p2
                diff_sum += (p1 - p2).sum().item()
                diff_sq_sum += torch.square(p1 - p2).sum().item()
                max_diff = max(max_diff, torch.max(diff).item())
                min_diff = min(min_diff, torch.min(diff).item())
                n += torch.numel(p1)
        diff_average = diff_sum / n
        L2_dist = torch.sqrt(torch.Tensor([diff_sq_sum])).item()
        return L2_dist, diff_average, diff_sum, max_diff, min_diff, n

    def interpolation_plot(self, dataset, buffer_dataloader):
        alpha_grid = np.arange(0, 1.001, 0.02)
        interpolation_accs = []
        for alpha in alpha_grid:
            net = self.deepcopy_model(self.net)
            old = self.deepcopy_model(self.old_model)
            new_model = interpolate(net, old, buffer_dataloader, self.device, alpha=alpha)
            acc, _, _ = evaluate(new_model, dataset, self.device)
            interpolation_accs.append(acc)
        print('interpolation accuracies:')
        print(interpolation_accs)
        print('old model accs:')
        accs, _, _ = evaluate(self.old_model, dataset, self.device)
        print(accs)
        print('new model accs:')
        accs, _, _ = evaluate(self.net, dataset, self.device)
        print(accs)

    def get_buffer_dataloder(self, batch_size=32):
        if batch_size is None:
            batch_size = len(self.buffer)
        buf_inputs, buf_labels = self.buffer.get_data(len(self.buffer), transform=self.transform)
        buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        buffer_dataloder = torch.utils.data.DataLoader(buffer_dataset, batch_size=batch_size, num_workers=0)
        return buffer_dataloder

    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy

    def train_model_after_interpolation(self, datalodaer):
        self.net.train()
        self.net = self.freeze_weights(self.net)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.0001)

        for input, target in datalodaer:
            optimizer.zero_grad()
            input = input.to(self.device)
            target = target.to(self.device)
            y_pred = self.net(input)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()

        self.net = self.unfreeze_weights(self.net)

    def freeze_weights(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not ('linear' in name or 'classifier' in name):
                param.requires_grad = False
        return model

    def unfreeze_weights(self, model: nn.Module):
        for _, param in model.named_parameters():
            param.requires_grad = True
        return model


@torch.no_grad()
def evaluate(network: ContinualModel, dataset, device, last=False, reduction='mean'):
    status = network.training
    network.eval()
    network.to(device)
    losses = []
    accs = []
    ns = []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, total = 0.0, 0.0
        loss = 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            loss += F.cross_entropy(outputs, labels, reduction=reduction)
            total += labels.shape[0]

        accs.append(correct / total * 100)
        losses.append(loss)
        ns.append(total)

    network.train(status)
    return accs, losses, ns
