import torch
from datetime import datetime

def get_datetime():
    # today = date.today()
    now = datetime.now()
    d1 = now.strftime("%Y%m%d-%H%M%S-")
    return d1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(parameters, cfg):
    if cfg.method == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                    lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters),
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters),
                                        lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == 'adadelta':
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, parameters),
                                         lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer
