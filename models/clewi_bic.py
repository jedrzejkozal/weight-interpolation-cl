import torch

from .bic import *
from .clewi_mixin import ClewiMixin


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='A bag of tricks for Continual learning.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--bic_epochs', type=int, default=250,
                        help='bias injector.')
    parser.add_argument('--temp', type=float, default=2.,
                        help='softmax temperature')
    parser.add_argument('--valset_split', type=float, default=0.1,
                        help='bias injector.')
    parser.add_argument('--multi_bic', type=int, default=0)
    parser.add_argument('--wd_reg', type=float, default=None,
                        help='bias injector.')
    parser.add_argument('--distill_after_bic', type=int, default=1)
    parser.add_argument('--interpolation_alpha', type=float, default=0.5,
                        help='interpolation alpha')
    parser.add_argument('--debug_interpolation', action='store_true')

    return parser


class ClewiBiC(ClewiMixin, BiC):
    NAME = 'clewi_bic'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.interpolation_alpha = args.interpolation_alpha
        self.old_model = self.deepcopy_model(backbone)

    def end_task(self, dataset):
        ClewiMixin.end_task(self, dataset)
        BiC.end_task(self, dataset)
