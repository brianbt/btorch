from pyrsistent import freeze
import torch
from torch import nn
import warnings

import re
from fnmatch import fnmatch
from typing import Tuple, List, Union, Dict, Iterable


def auto_gpu(model=None, parallel='auto', on=None):
    """turn model to gpu if possible and nn.DataParallel

    Args:
        parallel (str or bool): either ``auto``, True, False
          remember to increase batch size if using `parallel`
        on (List[int] or None): Only useful if using `parallel`
          if None -> use all available GPU
          else -> use only the gpu listed

    Returns:
        str: either ``cuda`` or ``cpu`` if no input arguments.
        (str, nn.Module): if have input arguments

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
        print("using GPU")
    else:
        device = 'cpu'
        print("using CPU")
    if model is None:
        return device
    model = model.to(device)
    if 'cuda' in device and parallel == 'auto' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=on)
    elif 'cuda' in device and parallel is True:
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
    This function only support setting a different learning rate for each parameter.
    Depending on the optimizer, you can set extra parameter for optmizer. See Notes

    Args:
        model (nn.Module): Pytorch Model
        base_lr (float): learning rate of all parameters
        groups (Dict[str, float]): key is `name` of layers, value is the `extra_lr` (or False).
          all layers that contains that `name` will have `lr` of base_lr*lr.
          Using fnmatch|regex to do check whether a layer contains that `name`.
          fnmatch is matching structure like `layer1*`, `layer?.conv?.`, `*conv2*`, etc...
          Hence, `name` here is either fnmatch or regex expression if using raw_query.
          regex is the comman regex matching.
          If `float` is False: those layers with `name` will be freeze. 
          In particular, not include in the return output and require_grad will be set to False
        ignore_the_rest (bool, optional): Include the final FC layers if True. Defaults to False.
        raw_query (bool, optional): Modify the keys of `groups` as f'*{key}*' if False
          Do not do any modification to the keys of `groups` if True. Defaults to False.
        regex (bool, optional): Use regex instead of fnmatch on keys of groups. Defaults to False.
          This will overrride raw_query to True. 
          regex=False is depracted

    Returns:
        List[Dict[str, Union[float, Iterable]]]: list of dict that has two or more key-value pair.
          The first one is feature generation layers. [those layers must start with `features` name] <usually is backbone>
            is a dict['params':list(model.parameters()), 'names':list(`layer's name`), 'query':query, 'lr':base_lr*groups[groups.keys()]]
          The second is all others layer. [all others params, if ignore_the_rest = False]
            is a dict['params':list(model.parameters()), 'names':list(`layer's name`), 'lr':base_lr]

    Examples:
        >>> model = models.resnet50()
        >>> # all layers that has name start with `layer1, layer2 and layer3` will have learning rate `0.001*0.01`
        >>> # all layers that has name start with `layer4` will have learning rate `0.001*0.001`
        >>> # for all other layers will have the base_lr `0.001`
        >>> model_params = finetune(model, base_lr=0.001, groups={'^layer[1-3].*': 0.01, '^layer4.*': 0.001}, regex=True)
        >>> # setting extra parameter (other than learning rate) for that optimizer
        >>> # the second param_group `layer4` will have weight_decay 1e-2
        >>> model_params[1]['weight_decay'] = 1e-2
        >>> # init optimizer with the above setting
        >>> optimizer = torch.optim.SGD(model_params, momentum=0.9, lr=0.1, weight_decay=5e-3)
    """
    if regex:
        raw_query = True
    else:
        warnings.warn("regex=False is deprecated; use regex=True", warnings.DeprecationWarning)
    # Deal with Freeze Later
    freeze_group = dict()
    for k,v in groups.items():
        if v is False:
            freeze_group[k] = 1
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
        if regex and re.match(freeze_group, k):
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
