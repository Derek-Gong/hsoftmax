import torch.optim as optim

OPTIMIZER_DICT = {'adam': optim.Adam, 'sgd': optim.SGD}

def init_optimizer(parameters, configs):
    assert configs['optim'] in OPTIMIZER_DICT
    optim = OPTIMIZER_DICT[configs['optim']]
    return optim(parameters, **configs['optim_conf'])