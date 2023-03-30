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

    # TODO: modify ClewiDer to recompute logits after each task - new tasks are to regularised
