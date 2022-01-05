import torch
import torch.nn.functional as F

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


def rolling_window(a, shape):
    """ Rolling window on np.array.

    Args:
        a (np.ndarray): Target array
        shape (tuple or int): the rolling windows size.
          if int -> roll the last dimension. (*,D) -> (*,D-shape+1,shape)
          if tuple -> roll the last len(tuple) dim. It works like conv.

    Returns:
        (np.ndarray): Rolled Array

    Usages:
        >>> # For 1D array
        >>> arr = np.arange(10)
        >>> # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> rolling_window(arr, 7)  #(4,7)
        >>> # array([[0, 1, 2, 3, 4, 5, 6],
        >>> #        [1, 2, 3, 4, 5, 6, 7],
        >>> #        [2, 3, 4, 5, 6, 7, 8],
        >>> #        [3, 4, 5, 6, 7, 8, 9]])
        >>> # For 2D array
        >>> arr = np.linspace((1,2,3,4,5),(16,17,18,19,20),4)
        >>> # array([[ 1.,  2.,  3.,  4.,  5.],
        >>> #        [ 6.,  7.,  8.,  9., 10.],
        >>> #        [11., 12., 13., 14., 15.],
        >>> #        [16., 17., 18., 19., 20.]])
        >>> arr = rolling_window(arr, 3) #(2, 3, 5)
        >>> # array([[[ 1.,  2.,  3.,  4.,  5.],
        >>> #         [ 6.,  7.,  8.,  9., 10.],
        >>> #         [11., 12., 13., 14., 15.]],
        >>> #         [[ 6.,  7.,  8.,  9., 10.],
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
    import pandas as pd
    import numpy as np
    if isinstance(a, pd.DataFrame):
        a = a.to_numpy()
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
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides).squeeze()