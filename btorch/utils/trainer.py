import os
import torch
from torch import nn
import numpy as np
import warnings

import re
from fnmatch import fnmatch
from typing import Tuple, List, Union, Dict, Iterable


class twoOptim():
    """Auto change the optimizer base on number on ``.step()`` called

    Args:
        optim1 (pytorch.optim): first optimizer
        optim2 (pytorch.optim): second optimizer
        change_step (int): number of ``.step()`` required to change optimizer

    Examples:
        >>> optim1 = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> optim2 = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-3)
        >>> optim = btorch.utils.trainer.twoOptim(optim1, optim2, 3)
        >>> optim.step()
    
    Note:
        This can be nested, you can wrap a ``twoOptim`` as ``optim1`` or ``optim2``
    """
    def __init__(self, optim1, optim2, change_step):
        self.optim1 = optim1
        self.optim2 = optim2
        self.change_step = change_step
        self.step_cnt = 0
        self.current = optim1
        self.changed = False

    def change_optim(self):
        if not self.changed:
            self.current = self.optim2
            self.changed = True

    def state_dict(self, *args, **kwargs):
        return self.current.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.current.load_state_dict(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.current.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.current.step(*args, **kwargs)
        self.step_cnt += 1
        if self.step_cnt == self.change_step:
            self.change_optim()

    def add_param_group(self, *args, **kwargs):
        self.current.add_param_group(*args, **kwargs)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += f'{self.optim1.__repr__()}'
        format_string += '\n'
        format_string += f'{self.optim2.__repr__()}'
        format_string += '\nUsing-----\n'
        format_string += f'{self.current.__repr__()}'
        format_string += '\n'
        format_string += ')'
        return format_string

def get_freer_gpu():
    """return the idx of the first available gpu"""
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def auto_gpu(model=None, parallel='auto', on=None):
    """turn model to gpu if possible and nn.DataParallel

    Args:
        parallel (str or bool): either ``auto``, ``True``, ``False``.
          Remember to increase batch size if using ``parallel``
        on (List[int] or None): Only useful if using ``parallel``.
          if None -> use all available GPU.
          else -> use only the gpu listed.

    Returns:
        str: either ``cuda`` or ``cpu`` if no input arguments.
        (str, nn.Module): if have input arguments.

    Examples:
        >>> device = auto_gpu()
        >>> device, model = auto_gpu(model)
        >>> device, model = auto_gpu(model, on=[0,2])
    """
    if torch.cuda.is_available():
        if on is not None and parallel is not False:
            device = f'cuda:{",".join(map(str, on))}'
        else:
            device = 'cuda'
        torch.backends.cudnn.benchmark = True
        print(f"auto_gpu: using GPU ({torch.cuda.get_device_name()})")
    else:
        device = 'cpu'
        print("auto_gpu: using CPU")
    if model is None:
        return device
    model = model.to(device)
    if 'cuda' in device and parallel == 'auto' and torch.cuda.device_count() > 1:
        print(f"auto_gpu: using nn.DataParallel on {torch.cuda.device_count()}GPU, consider increase batch size {torch.cuda.device_count()} times")
        model = nn.DataParallel(model, device_ids=on)
    elif 'cuda' in device and parallel is True:
        print("auto_gpu: using nn.DataParallel, consider increase batch size")
        model = nn.DataParallel(model, device_ids=on)
    return device, model

def finetune(
        model: nn.Module,
        base_lr: float,
        groups: Dict[str, float],
        ignore_the_rest: bool = False,
        raw_query: bool = False,
        regex=False) -> List[Dict[str, Union[float, Iterable]]]:
    """ This is something call per-parameter options

    Separate out the finetune parameters with a learning rate for each layers of parameters
    This function only support setting a different learning rate for each layer's arameter.
    Depending on the optimizer, you can set extra parameter for that layer for the optmizer -> See Notes 
    If you freeze layer using this function and want to unfreeze it later:
    See https://discuss.pytorch.org/t/correct-way-to-freeze-layers/26714/2

    Args:
        model (nn.Module): Pytorch Model.
        base_lr (float): learning rate of all layers.
        groups (Dict[str, float]): key is ``name`` of layers, value is the ``extra_lr`` (or False).
          all layers that contain that ``name`` will have ``lr`` of base_lr*extra_lr.
          it uses fnmatch|regex to check whether a layer contains that ``name``.
          fnmatch is matching structure like ``layer1*``, ``layer?.conv?.``, ``*conv2*``, etc...
          Regex is the comman regex matching.
          Hence, ``name`` here is either fnmatch or regex expression if using raw_query.
          If ``float`` is False: those layers with ``name`` will be frozen.
          In particular, they will not be included in the return output and require_grad will be set to False.
        ignore_the_rest (bool, optional): Include the remaining layer that are not stated in ``grouprs`` or not. Defaults to False.
        raw_query (bool, optional): Modify the keys of ``groups`` as f'*{key}*' if False. Only useful when ``regex=False``
          Do not do any modification to the keys of ``groups`` if True. Defaults to False.
        regex (bool, optional): Deprecated when ``regex=False``. Use regex instead of fnmatch on keys of groups. Defaults to False.
          This will overrride raw_query to True if set to True.

    Note:
     ``regex=False`` is depracted.

    Returns:
        List[Dict[str, Union[float, Iterable]]]: list of dict that has two or more key-value pair.
          The first one is feature generation layers. [those layers must start with ``features`` name] <usually is backbone> is a
            ``dict['params':list(model.parameters()), 'names':list(`layer's name`), 'query':query, 'lr':base_lr*groups[groups.keys()]]``.
          The remaining are all others layer. [all others params for last one, if ignore_the_rest = False] is a
            ``dict['params':list(model.parameters()), 'names':list(`layer's name`), 'lr':base_lr]``.

    Examples:
        >>> model = models.resnet50()
        >>> # all layers that has name start with ``layer1 and layer2`` will have learning rate ``0.001*0.01``
        >>> # all layers that has name start with ``layer3`` will be froozen``
        >>> # all layers that has name start with ``layer4`` will have learning rate ``0.001*0.001``
        >>> # for all other layers will have the base_lr ``0.001``
        >>> model_params = finetune(model, base_lr=0.001, groups={'^layer[1-2].*': 0.01, '^layer3.*': False, '^layer4.*': 0.001}, regex=True)
        >>> # setting extra parameter (other than learning rate) for that optimizer
        >>> # the second param_group ``layer4`` will have weight_decay 1e-2
        >>> model_params[1]['weight_decay'] = 1e-2
        >>> # init optimizer with the above setting
        >>> # the argument under ``torch.optim.SGD`` will be overrided by finetune() if they exist.
        >>> # For example, all model_params will have weight_decay=5e-3 except model_params[1]
        >>> optimizer = torch.optim.SGD(model_params, momentum=0.9, lr=0.1, weight_decay=5e-3)
    """
    if regex:
        raw_query = True
    else:
        warnings.warn("regex=False is deprecated; use regex=True", DeprecationWarning)
    # Deal with Freeze Later
    freeze_group = dict()
    freeze = False
    for k,v in groups.items():
        if v is False:
            freeze_group[k] = 1
            freeze=True
    for k in freeze_group.keys():
        del groups[k]
    freeze_group = "(" + ")|(".join(freeze_group) + ")"

    parameters = [
        dict(params=[],
             names=[],
             query=query if raw_query else '*' + query + '*',
             lr = lr * base_lr,
             initial_lr = lr * base_lr) for query, lr in groups.items()
    ]
    rest_parameters = dict(params=[], names=[], lr=base_lr, initial_lr=base_lr)
    for k, v in model.named_parameters():
        rest = 0
        if freeze and regex and re.match(freeze_group, k):
            v.requires_grad = False
            continue
        for group in parameters:
            if not regex and fnmatch(k, group['query']):
                group['params'].append(v)
                group['names'].append(k)
                rest = 1
                break
            elif regex and re.compile(group['query']).search(k):
                group['params'].append(v)
                group['names'].append(k)
                rest = 1
                break
        if rest == 0:
            rest_parameters['params'].append(v)
            rest_parameters['names'].append(k)

    if not ignore_the_rest:
        parameters.append(rest_parameters)
    for group in parameters:
        group['params'] = iter(group['params'])
    return parameters


def freeze(model):
    """Freeze all layers of a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    """Freeze all layers of a model."""
    for param in model.parameters():
        param.requires_grad = True

def L1Regularizer(model, lambda_=1e-4):
    """
    Add L1 regularization to the model. Notice: weight_decay is L2 reg.

    Examples:
        >>> optimizer.zero_grad()
        >>> predicted = net(inputs)
        >>> loss = criterion(predicted, targets)
        >>> loss += L1Regularizer(net)  ## add L1 regularization
        >>> loss.backward()
        >>> optimizer.step()
    """
    return lambda_*sum(p.norm(p=1) for p in model.parameters() if p.requires_grad)
