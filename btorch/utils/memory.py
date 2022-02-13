# This is an improved version of the following function
# https://github.com/Oldpan/Pytorch-Memory-Utils
import gc
import datetime
import inspect

import torch
from torch import nn
import numpy as np

dtype_memory_size_dict = {  #these are in bytes
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}
bytes_to = {
    'bit': 1/8,
    'byte': 1,
    'bytes': 1,
    'b': 1,
    'kb': 1000,
    'mb': 1000 * 1000,
    'gb': 1000 * 1000 * 1000,
}
# compatibility of torch1.0
if getattr(torch, "bfloat16", None) is not None:
    dtype_memory_size_dict[torch.bfloat16] = 16/8
if getattr(torch, "bool", None) is not None:
    dtype_memory_size_dict[torch.bool] = 8/8 # pytorch use 1 byte for a bool, see https://github.com/pytorch/pytorch/issues/41571

def get_mem_space(x):
    try:
        ret = dtype_memory_size_dict[x]
    except KeyError:
        print(f"dtype {x} is not supported!")
    return ret

class MemTracker(object):
    """
    Class used to track pytorch memory usage

    Args:
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file. Defaults to None, which means will not output to a log file.
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0

    Examples:
        >>> gpu_tracker = MemTracker()              # define a GPU tracker
        >>> gpu_tracker.track()                     # run function between the code line where uses GPU
        >>> cnn = models.vgg19(pretrained=True).features.to('cuda').eval()
        >>> gpu_tracker.track()                     # run function between the code line where uses GPU
        >>> gpu_tracker.done()                      # call this function to print the result
    """
    def __init__(self, detail=True, path=None, verbose=False, device=0):
        self.print_detail = detail
        self.last_tensor_sizes = set()
        if path is not None:
            self.gpu_profile_fn = path + f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt'
        else:
            self.gpu_profile_fn = None
        self.verbose = verbose
        self.begin = True
        self.device = device
        self.outstr = ""

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def get_tensor_usage(self):
        sizes = [np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype) for tensor in self.get_tensors()]
        return np.sum(sizes) / 1024**2

    def get_allocate_usage(self):
        return torch.cuda.memory_allocated() / 1024**2

    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def print_all_gpu_tensor(self, file=None):
        for x in self.get_tensors():
            print(x.size(), x.dtype, np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2, file=file)

    def done(self):
        print(self.outstr)

    def track(self):
        """
        Track the GPU memory usage
        """
        frameinfo = inspect.stack()[1]
        where_str = frameinfo.filename + ' line ' + str(frameinfo.lineno) + ': ' + frameinfo.function
        if self.begin:
            self.outstr+=(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                    f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                    f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")
            self.begin = False

        if self.print_detail is True:
            ts_list = [(tensor.size(), tensor.dtype) for tensor in self.get_tensors()]
            new_tensor_sizes = {(type(x),
                                tuple(x.size()),
                                ts_list.count((x.size(), x.dtype)),
                                np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2,
                                x.dtype) for x in self.get_tensors()}
            for t, s, n, m, data_type in new_tensor_sizes - self.last_tensor_sizes:
                self.outstr+=(f'+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')
            for t, s, n, m, data_type in self.last_tensor_sizes - new_tensor_sizes:
                self.outstr+=(f'- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')

            self.last_tensor_sizes = new_tensor_sizes

        self.outstr+=(f"\n↳At {where_str:<50}"
                f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")


        if self.gpu_profile_fn is not None:
            with open(self.gpu_profile_fn, 'a+') as f:

                if self.begin:
                    f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                            f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                            f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")
                    self.begin = False

                if self.print_detail is True:
                    ts_list = [(tensor.size(), tensor.dtype) for tensor in self.get_tensors()]
                    new_tensor_sizes = {(type(x),
                                        tuple(x.size()),
                                        ts_list.count((x.size(), x.dtype)),
                                        np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2,
                                        x.dtype) for x in self.get_tensors()}
                    for t, s, n, m, data_type in new_tensor_sizes - self.last_tensor_sizes:
                        f.write(f'+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')
                    for t, s, n, m, data_type in self.last_tensor_sizes - new_tensor_sizes:
                        f.write(f'- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')

                    self.last_tensor_sizes = new_tensor_sizes

                f.write(f"\n↳At {where_str:<50}"
                        f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                        f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")

def memory_summary(device=None, return_unit='bytes'):
    """Print the memory usage of the current device.
    
    Args:
        device (int, optional): see https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html
    """
    print(f"GPU Memory Summary | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |")
    print(f'{torch.cuda.memory_allocated(device)/bytes_to[return_unit.lower()]:.2f} {return_unit} USED')
    print(f'{torch.cuda.memory_reserved(device)/bytes_to[return_unit.lower()]:.2f} {return_unit} RESERVED')
    print(f'{torch.cuda.max_memory_allocated(device)/bytes_to[return_unit.lower()]:.2f} {return_unit} USED MAX')
    print(f'{torch.cuda.max_memory_reserved(device)/bytes_to[return_unit.lower()]:.2f} {return_unit} RESERVED MAX')

def tensor_size(tensor, return_unit='bytes'):
    """input torch.tensor, return the theroteical size"""
    if return_unit.lower() not in bytes_to:
        raise ValueError(f"return_unit should be one of {list(bytes_to.values())}. Got '{return_unit}'")
    return np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype) / bytes_to[return_unit.lower()]

def model_size(model, input, type_size=4):
    """given a model and input, return the model size
    
    Warnings:
        This function may not work well for complicated model.
        It only works for model that has layer by layer defined in the same order stated in forward.
        It may not work for model that has nested layer.
        It may not work for resnet50 those type of model.

    Args:
        model (nn.Module): PyTorch Model
        input (Tensor or list of size): 
          if Tensor -> should be able to call `model(input)`
          else -> size of input
        type_size (int, optional): dtype of input, Default to 4 (float32)
    """
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    if isinstance(input, (tuple, list, int)):
        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        input = torch.randn(input, device=device)
    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out
    print(out_sizes)
    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums
    print('Model {} : intermedite variables: {:3f} Mb (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} Mb (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


def del_gpu(ls=None, release_gpu=True):
    """Delete objects and release GPU memory"""
    if ls is not None:
        if isinstance(ls, (list, tuple)):
            for i in ls:
                del (i)
        else:
            del (ls)
    if release_gpu:
        gc.collect()
        torch.cuda.empty_cache()
       