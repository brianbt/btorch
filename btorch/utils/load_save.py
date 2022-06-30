import os
import warnings
import torch

def save_model(model, path, extra=None, optimizer=None, lr_scheduler=None):
    """torch.save enhanced. Will auto handle nn.DataParallel(model)

    Args:
        model (nn.Module): pytorch model
        path (str): saving_path
        extra (dict, optional): 
          Extra things that want to save. 
          Must reserve key ``model``.
          Defaults to None.
        optimizer (torch.optim, optional): pytorch optimizer.
          You can also put optim.state_dict() under ``extra`` instead of using this arg.
          It accepts list or dict of optimizer.
        lr_scheduler (torch.optim, optional): pytorch lr schedular.
          You can also put lr_s.state_dict() under ``extra`` instead of using this arg.
    """
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    if extra is not None:
        to_save = extra.copy()
    else:
        to_save = dict()
    if 'model' in to_save:
        raise Exception("``extra`` should not contains key ``model``")
    to_save['model'] = state_dict
    if optimizer is not None:
        if 'optimizer' in to_save:
            warnings.warn("key ``optimizer`` is already in ``extra``, replacing the ``optimizer``")
        if isinstance(optimizer, dict):
             to_save['optimizer'] = dict()
             for k in optimizer.keys():
                to_save['optimizer'][k] = optimizer[k].state_dict()
        elif isinstance(optimizer, list):
            to_save['optimizer'] = []
            for optim in optimizer:
                to_save['optimizer'].append(optim.state_dict())
        else:
            to_save['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        if 'lr_scheduler' in to_save:
            warnings.warn("key ``lr_scheduler`` is already in ``extra``, replacing the ``lr_scheduler``")
        to_save['lr_scheduler'] = lr_scheduler.state_dict()
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if 'epoch' not in to_save:
        warnings.warn("``epoch`` is not in ``extra``, consider adding it to keep track of which epoch you are saving.")
    torch.save(to_save, path)


def resume(path, model, optimizer=None, lr_scheduler=None):
    """Load all components for resume training. It load everything by reference

    Args:
        path (str): Load path. Must contains ['model'] and ['optimizer']
        model (nn.Module): Pytorch Model
        optimizer (torch.optim): Pytorch Optimizer
          It accepts list or dict of optimizer.
          Note that the order of list must be same as the order when you save it.
        lr_scheduler (torch.optim): Pytorch Lr Scheduler

    Returns:
        int: epoch (use it as start_epoch)
    """
    state = torch.load(path)
    epoch = state['epoch'] if 'epoch' in state else len(state['train_loss_data'])
    model.load_state_dict(state['model'])
    if optimizer is not None:
        if isinstance(optimizer, dict):
             for k in state['optimizer'].keys():
                optimizer[k].load_state_dict(state['optimizer'][k])
        elif isinstance(optimizer, list):
            for i in range(len(optimizer)):
                 optimizer[i].load_state_dict(state['optimizer'][i])
        else:
            optimizer.load_state_dict(state['optimizer'])
    if lr_scheduler is not None:
        if 'lr_scheduler' in state:
            for i in range(state['lr_scheduler']['last_epoch'], epoch, state['lr_scheduler']['_step_count']):
                lr_scheduler.step()
        else:
            for i in range(epoch):
                lr_scheduler.step()
        if 'lr_scheduler' in state:
            lr_scheduler.load_state_dict(state['lr_scheduler'])
    print(f"Loaded (by reference) at epoch {epoch}.")
    return epoch
