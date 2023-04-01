import torch

from .der import *
from .clewi_mixin import ClewiMixin


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning with Weight Interpolation with DER')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--interpolation_alpha', type=float, default=0.5,
                        help='interpolation alpha')
    return parser


class ClewiDer(Der, ClewiMixin):
    NAME = 'clewi_der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.interpolation_alpha = args.interpolation_alpha
        self.old_model = self.deepcopy_model(backbone)

    def end_task(self, dataset):
        """recompute logits after each task for less limitation on training with new tasks"""
        ClewiMixin.end_task(self, dataset)
        with torch.no_grad():
            data = self.buffer.get_data(len(self.buffer), transform=self.transform)
            buf_inputs = data[0]
            outputs = self.net(buf_inputs)
            self.buffer.logits = outputs.data
