import warnings
import numpy as np
import random
import torch
import torch.nn.functional as F

def to_tensor(a):
    """Turns any object into a tensor.
    """
    import pandas as pd
    if isinstance(a, pd.DataFrame):
        a = torch.tensor(a.to_numpy())
    if isinstance(a, list) and isinstance(a[0], np.ndarray):
        a = torch.stack([torch.tensor(x) for x in a])
    if isinstance(a, list) and isinstance(a[0], torch.Tensor):
        a = torch.stack(a)
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    return a

def seed_everythin(seed):
    """set seed on everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def change_batch_size(loader, batch_size):
    """Change the batch_size of a dataloader

    Args:
        loader (torch.utils.data.dataloader): Pytorch DataLoader
        batch_size (int): new batch_size

    Returns:
        torch.utils.data.dataloader
    """
    tmp = loader.__dict__
    tmp = {k: v for k, v in tmp.items() if (
        k[0] != "_" and k != "batch_sampler")}
    tmp['batch_size'] = batch_size
    return torch.utils.data.DataLoader(**tmp)

def rolling_window(a, shape, stride=1, step=1, dim=0, safe_check=False):
    """Rolling window for Tensor

    Returns a *view* of the original tensor when *step=1*, otherwise returns a *new* tensor.
    Only maintain when `a` is 1D or 2D.

    Args:
        a (Tensor): Pytorch Tensor
        shape (int): Window Size
        stride (int, optional): 
          stride within Window, ONLY useful when dim=-1. 
          If dm!=-1, stride is same as step. For example, if stride and step are both 2, the final step size will be 2*2=4.
          Defaults to 1.
        step (int, optional):
          select every `step` slices after rolling window.
          Defaults to 1.
        dim (int, optional): dimension in which rolling window happens. Defaults to 0.
        safe_check (bool, optional): `a` can be numpy array, Pandas, list, etc. This may affect performance.

    Returns:
        Tensor: PyTorch Tensor

    Examples:
        >>> # For 1D case
        >>> arr = torch.arange(10, dtype=torch.float32)
        >>> tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> rolling_window(arr, 3)
        >>> tensor([[0., 1., 2.],
        >>>         [1., 2., 3.],
        >>>         [2., 3., 4.],
        >>>         [3., 4., 5.],
        >>>         [4., 5., 6.],
        >>>         [5., 6., 7.],
        >>>         [6., 7., 8.],
        >>>         [7., 8., 9.]])
        >>> rolling_window(arr, 4, 2, 2)
        >>> tensor([[0., 2.],
        >>>         [2., 4.],
        >>>         [4., 6.],
        >>>         [6., 8.]])
        >>> # For 2D case
        >>> arr = torch.arange(1, 21, dtype=torch.float32).view(4,-1)
        >>> tensor([[ 1.,  2.,  3.,  4.,  5.],
        >>>         [ 6.,  7.,  8.,  9., 10.],
        >>>         [11., 12., 13., 14., 15.],
        >>>         [16., 17., 18., 19., 20.]])
        >>> rolling_window(arr, 2, 1, 1)
        >>> tensor([[[ 1.,  2.,  3.,  4.,  5.],
        >>>          [ 6.,  7.,  8.,  9., 10.]],
        >>>         [[ 6.,  7.,  8.,  9., 10.],
        >>>          [11., 12., 13., 14., 15.]],
        >>>         [[11., 12., 13., 14., 15.],
        >>>          [16., 17., 18., 19., 20.]]])
        >>> rolling_window(arr, 2, 1, 2)
        >>> tensor([[[ 1.,  2.,  3.,  4.,  5.],
        >>>          [ 6.,  7.,  8.,  9., 10.]],
        >>>         [[11., 12., 13., 14., 15.],
        >>>          [16., 17., 18., 19., 20.]]])
        >>> rolling_window(arr, 4, 2, 1, 1)
        >>> tensor([[[ 1.,  3.],
        >>>          [ 2.,  4.]],
        >>>         [[ 6.,  8.],
        >>>          [ 7.,  9.]],
        >>>         [[11., 13.],
        >>>          [12., 14.]],
        >>>         [[16., 18.],
        >>>          [17., 19.]]])
    """
    if safe_check:
        a = to_tensor(a)
    if a.dim()-dim == 1:
        shape = a.shape[-1] - shape + 1
    out = a.unfold(dim, shape, stride).transpose(-2, -1)
    if step != 1:
        out = torch.index_select(out, dim, torch.arange(0, out.shape[dim], step))
    return out

def conv_window2d(a, window, stride=None):
    """Convolution like rolling windows for 2D

    Args:
        a (Tensor): Tensor. Support (N, C, H, W) and (H, W)
        window (int or Tuple(int)): kernel size
        stride (int, optional): Defaults to `window`
    """
    batch_mode = True
    if a.dim() == 2:
        a = a.unsqueeze(0).unsqueeze(0)
        batch_mode = False
    elif a.dim() != 4:
        raise ValueError("Only support (N, C, H, W) and (H, W)")
    if isinstance(window, int):
        kernel_h, kernel_w = window, window
    else:
        kernel_h, kernel_w = window[-2], window[-1]
    stride = (kernel_h, kernel_w) if stride is None else stride
    if isinstance(stride, int):
        stride = (stride, stride)
    patches = a.unfold(2, kernel_h, stride[0]).unfold(3, kernel_w, stride[1])
    if not batch_mode:
        patches = patches.squeeze(0).squeeze(0)
    return patches

def adaptive_conv_window(a, shape, stride=1, dim=0):
    """ Rolling window on np.array.

    This function is not well developed yet. This function is slow.

    Args:
        a (np.ndarray): Target array
        shape (tuple or int): the output size.
          if int -> For ND -> roll the last dimension. (*,D) -> (*,D-shape+1,shape)
                 -> For 2D -> roll the second dimension. (N,D) -> (N-shape+1,shape,D) [use for time series]
          if tuple -> roll the last len(tuple) dim. It works like conv.
        stride (int, optional): timestep for rolling on first dim. [only meaning fulling when 1D or 2D]
        dim (int, optional): dimension in which rolling window happens. Defaults to 0.

    Returns:
        (np.ndarray): Rolled Array

    Notes:
        use `rolling_window()` when dealing with 1D or 2D time series data, 
        `rolling_window()` is roll the entire dimension.
        This function is a roll like a convolution.

    Usages:
        >>> # For 1D array
        >>> arr = np.arange(10) #(10,)
        >>> # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> rolling_window(arr, 7)  #(4,7)
        >>> arr.unfold(0, 4, 1).transpose(1, 0)     #SAME results 
        >>> # array([[0, 1, 2, 3, 4, 5, 6],
        >>> #        [1, 2, 3, 4, 5, 6, 7],
        >>> #        [2, 3, 4, 5, 6, 7, 8],
        >>> #        [3, 4, 5, 6, 7, 8, 9]])
        >>> # For 2D array
        >>> arr = np.linspace((1,2,3,4,5),(16,17,18,19,20),4) #(4,5)
        >>> # array([[ 1.,  2.,  3.,  4.,  5.],
        >>> #        [ 6.,  7.,  8.,  9., 10.],
        >>> #        [11., 12., 13., 14., 15.],
        >>> #        [16., 17., 18., 19., 20.]])
        >>> arr = rolling_window(arr, 3) #(2, 3, 5)
        >>> arr.unfold(0, 3, 1).transpose(2,1)      #SAME results
        >>> # array([[[ 1.,  2.,  3.,  4.,  5.],
        >>> #         [ 6.,  7.,  8.,  9., 10.],
        >>> #         [11., 12., 13., 14., 15.]],
        >>> #        [[ 6.,  7.,  8.,  9., 10.],
        >>> #         [11., 12., 13., 14., 15.],
        >>> #         [16., 17., 18., 19., 20.]]])
        >>> # For 3D array
        >>> arr = rolling_window(arr, (2,2,2,5)) #(2, 2, 2, 2, 5)
        >>> # array([[[[[ 1.,  2.,  3.,  4.,  5.],
        >>> #           [ 6.,  7.,  8.,  9., 10.]],
        >>> #          [[ 6.,  7.,  8.,  9., 10.],
        >>> #           [11., 12., 13., 14., 15.]]],
        >>> #         [[[ 2.,  3.,  4.,  5.,  6.],
        >>> #           [ 7.,  8.,  9., 10., 11.]],
        >>> #          [[ 7.,  8.,  9., 10., 11.],
        >>> #           [12., 13., 14., 15., 16.]]]],
        >>> #        [[[[ 6.,  7.,  8.,  9., 10.],
        >>> #           [11., 12., 13., 14., 15.]],
        >>> #          [[11., 12., 13., 14., 15.],
        >>> #           [16., 17., 18., 19., 20.]]],
        >>> #         [[[ 7.,  8.,  9., 10., 11.],
        >>> #           [12., 13., 14., 15., 16.]],
        >>> #          [[12., 13., 14., 15., 16.],
        >>> #           [17., 18., 19., 20.,  1.]]]]])

    """
    warnings.warn("rolling_window_old() is deprecated; use rolling_window()", DeprecationWarning)
    import pandas as pd
    import numpy as np
    if isinstance(a, pd.DataFrame):
        a = a.to_numpy()
    if isinstance(a, torch.Tensor):
        a = a.numpy()
    # Args Check
    if not isinstance(shape, int) and len(shape) == 1:
        shape = shape[0]
    if len(a.shape) == 2 and isinstance(shape, int):
        shape = (shape, a.shape[-1])
    if isinstance(shape, int):    # rolling window last dim
        s = a.shape[:-1] + (a.shape[-1] - shape + 1, shape)
        strides = a.strides + (a.strides[-1],)
    else:   # rolling window for any dim
        s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
        strides = a.strides + a.strides
    out = torch.tensor(np.lib.stride_tricks.as_strided(a, shape=s, strides=strides).squeeze())
    return out[::stride]

def accuracy_score(y_pred, y, normalize=True, sample_weight=None):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true=y, y_pred=y_pred, normalize=normalize, sample_weight=sample_weight)
