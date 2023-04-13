import torch
import torch.utils.data
import copy

from utils.buffer import Buffer
from models.utils.weight_interpolation import *


class ClewiMixin:

    def end_task(self, dataset):
        buffer_dataloder = self.get_buffer_dataloder()
        torch.save(self.old_model, 'old_model.pt')
        torch.save(self.net, 'net.pt')

        self.old_model = interpolate(self.net, self.old_model, buffer_dataloder, self.device, alpha=self.interpolation_alpha)
        self.net = self.deepcopy_model(self.old_model)
        self.opt = self.opt.__class__(self.net.parameters(), **self.opt.defaults)
        self.opt.zero_grad()

    def get_buffer_dataloder(self):
        data = self.buffer.get_data(len(self.buffer), transform=self.transform)
        buf_inputs = data[0]
        buf_labels = data[1]

        buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        buffer_dataloder = torch.utils.data.DataLoader(buffer_dataset, batch_size=32, num_workers=0)
        return buffer_dataloder

    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        model_copy.load_state_dict(model.state_dict())
        return model_copy
