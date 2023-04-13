import torch
import torch.nn.functional as F
import copy

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Online Continual Learning with Maximally Interfered Retrieval')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Mir(ContinualModel):
    NAME = 'mir'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        assert args.minibatch_size > args.batch_size

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)

            grad_dims = []
            for param in self.net.parameters():
                grad_dims.append(param.data.numel())
            grad_vector = self.get_grad_vector(self.net.parameters, grad_dims)
            model_tmp = self.get_future_step_parameters(self.net, grad_vector, grad_dims, lr=self.opt.defaults['lr'])

            with torch.no_grad():
                y_hat_pre = self.net(buf_inputs)
                pre_loss = F.cross_entropy(y_hat_pre, buf_labels, reduction='none')

                y_hat_post = model_tmp(buf_inputs)
                post_loss = F.cross_entropy(y_hat_post, buf_labels, reduction="none")

                scores = post_loss - pre_loss
                idx = scores.sort(descending=True)[1][:self.args.batch_size]

            buf_inputs = buf_inputs[idx]
            buf_labels = buf_labels[idx]

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def get_grad_vector(self, pp, grad_dims):
        """
        gather the gradients in one vector
        """
        grads = torch.Tensor(sum(grad_dims))
        grads = grads.to(self.device)

        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads

    def get_future_step_parameters(self, this_net, grad_vector, grad_dims, lr=1):
        new_net = copy.deepcopy(this_net)
        self.overwrite_grad(new_net.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_net.parameters():
                if param.grad is not None:
                    param.data = param.data - lr*param.grad.data
        return new_net

    def overwrite_grad(self, parameters, new_grad, grad_dims):
        cnt = 0
        for param in parameters():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1
