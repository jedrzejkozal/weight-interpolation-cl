import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import copy

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer_pseudorehersal import BufferPseudoRehersal
from models.utils.weight_interpolation import *

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning with Weight Interpolation')
    parser.add_argument('--interpolation_alpha', type=float, default=0.5,
                        help='interpolation alpha')
    parser.add_argument('--permuation_epochs', type=int, default=1)
    parser.add_argument('--batchnorm_epochs', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=10000, help='number of images sampled from pseudorehersal used for activations compuation during interpolation')

    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class ClewiPseudorehersal(ContinualModel):
    NAME = 'clewi_pseudorehersal'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = BufferPseudoRehersal(self.device)
        self.old_model = self.deepcopy_model(backbone)
        self.interpolation_alpha = args.interpolation_alpha

        self.first_task = True


    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        # if not self.buffer.is_empty():
        #     buf_inputs, buf_labels = self.buffer.get_data(
        #         self.args.minibatch_size, transform=self.transform)
        #     inputs = torch.cat((inputs, buf_inputs))
        #     labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
    
    def end_task(self, dataset):
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
            return

        buffer_dataloder = self.get_buffer_dataloder()
        self.old_model = interpolate(self.net, self.old_model, buffer_dataloder, self.device,
                                     alpha=self.interpolation_alpha, permuation_epochs=self.args.permuation_epochs, batchnorm_epochs=self.args.batchnorm_epochs)
        # self.train_model_after_interpolation(buffer_dataloder)
        self.net = self.deepcopy_model(self.old_model)
        self.opt = self.opt.__class__(self.net.parameters(), **self.opt.defaults)
        self.opt.zero_grad()

        # torch.save(self.old_model, 'old_model.pt')
        # torch.save(self.net, 'net.pt')

        # self.interpolation_plot(dataset, buffer_dataloder)

    def get_buffer_dataloder(self):
        buf_inputs, buf_labels = self.buffer.get_data(self.args.sample_size)
        buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        buffer_dataloder = torch.utils.data.DataLoader(buffer_dataset, batch_size=128, num_workers=0)
        return buffer_dataloder

    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        model_copy.load_state_dict(model.state_dict())
        return model_copy