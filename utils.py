import random
import os
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_model(model, name):
    state_dict = model.state_dict()
    torch.save(state_dict, f'weights/{name}.pt')


def load_model(model, name):
    # state_dict = torch.load(f'weights/{name}.pt')
    state_dict = torch.load(f'{name}.pt')
    model.load_state_dict(state_dict)
