import warnings
import numpy as np
import random
import torch
import torch.nn.functional as F
from packaging.version import parse as _parse
from torch.utils.data import TensorDataset


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


def seed_everything(seed):
    """set seed on everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_differentiable(func, *args, **kwargs):
    """Check if a function is differentiable.
      """
    try:
        func(torch.ones(1, requires_grad=True), *args, **kwargs).grad_fn
        return True
    except Exception as e:
        print('error message:', e)
        return False


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
      Only maintain when ``a`` is 1D or 2D.

      Args:
          a (Tensor): Pytorch Tensor
          shape (int): Window Size
          stride (int, optional):
            stride within Window, ONLY useful when dim=-1.
            If ``dm``!=-1, stride is same as step. For example, if stride and step are both 2, the final step size will be 2*2=4.
            Defaults to 1.
          step (int, optional):
            select every ``step`` slices after rolling window.
            Defaults to 1.
          dim (int, optional): dimension in which rolling window happens. Defaults to 0.
          safe_check (bool, optional): ``a`` can be numpy array, Pandas, list, etc. This may affect performance.

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
    if a.dim() - dim == 1:
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
          stride (int, optional): Defaults to ``window``
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
          use ``rolling_window()`` when dealing with 1D or 2D time series data,
          ``rolling_window()`` is roll the entire dimension.
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
    if isinstance(shape, int):  # rolling window last dim
        s = a.shape[:-1] + (a.shape[-1] - shape + 1, shape)
        strides = a.strides + (a.strides[-1],)
    else:  # rolling window for any dim
        s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
        strides = a.strides + a.strides
    out = torch.tensor(np.lib.stride_tricks.as_strided(a, shape=s, strides=strides).squeeze())
    return out[::stride]


def accuracy_score(y_pred, y, normalize=True, sample_weight=None):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true=y, y_pred=y_pred, normalize=normalize, sample_weight=sample_weight)


# https://discuss.pytorch.org/t/nested-list-of-variable-length-to-a-tensor/38699/21?u=brianbt
def ints_to_tensor(ints, pad=0):
    """Converts a nested list of integers to a padded tensor, with padding value ``pad``

      Args:
          ints (list[]): A nested list of integers or Tensor
          pad (int, optional): The padding value. Defaults to 0.

      Note:
          If you need to limit the length of the dimension, do ``output[:, :max_len]``.
      """
    if isinstance(ints, torch.Tensor):
        return ints
    if isinstance(ints, list):
        if isinstance(ints[0], int):
            return torch.LongTensor(ints)
        if isinstance(ints[0], torch.Tensor):
            return _pad_tensors(ints, pad)
        if isinstance(ints[0], list):
            return ints_to_tensor([ints_to_tensor(inti) for inti in ints], pad)


def _pad_tensors(tensors, pad):
    """helper function of ``ints_to_tensor()``
      Takes a list of ``N`` M-dimensional tensors (M<4) and returns a padded tensor.

      The padded tensor is ``M+1`` dimensional with size ``N, S1, S2, ..., SM``
      where ``Si`` is the maximum value of dimension ``i`` amongst all tensors.
      """
    rep = tensors[0]
    padded_dim = []
    for dim in range(rep.dim()):
        max_dim = max([tensor.size(dim) for tensor in tensors])
        padded_dim.append(max_dim)
    padded_dim = [len(tensors)] + padded_dim
    padded_tensor = torch.zeros(padded_dim) + pad
    padded_tensor = padded_tensor.type_as(rep)
    for i, tensor in enumerate(tensors):
        size = list(tensor.size())
        if len(size) == 1:
            padded_tensor[i, :size[0]] = tensor
        elif len(size) == 2:
            padded_tensor[i, :size[0], :size[1]] = tensor
        elif len(size) == 3:
            padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
        else:
            raise ValueError('Padding is supported for upto 3D tensors at max.')
    return padded_tensor


def number_params(model, exclude_freeze=False):
    """calculate the number of parameters in a model

      Args:
          model (nn.Module): PyTorch model
          exclude_freeze (bool, optional): Whether to count the frozen layer. Defaults to False.
      """
    pp = 0
    for p in list(model.parameters()):
        if exclude_freeze and p.requires_grad is False:
            continue
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def digit_version(version_str, length=4):
    """Convert a version string into a tuple of integers.

      This method is usually used for comparing two versions.
      ``digit_version("1.6.7") > digit_version("1.8")`` will return False.
      For pre-release versions: alpha < beta < rc.

      Args:
          version_str (str): The version string.
          length (int): The maximum number of version levels. Default: 4.

      Returns:
          tuple[int]: The version info in digits (integers).
      """
    version = _parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


def get_class_weight(a, method='sklearn', total_nu_class=None):
    """For auto generate class weight for imbalanced dataset. Only support multi-class classification.

      Note:
          Are you looking for https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

      Args:
          a (list or Tensor): The target labels. This should be your ``y_train``.
          method (str, optional): See Examples.
            Support either ['sklearn','inverse_size', 'inverse_sqrt_size', 'inverse_proba', 'inverse_sqrt_proba', 'inverse_softmax', 'inverse_sqrt_softmax']
            Defaults to `'sklearn' <https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html>`__.
          total_nu_class (int, optional): Total number of class.
            If ``a`` does not contain all possible class label, you should set this parameter.
            Assume the first class is 0, the second class is 1, and so on.
            Defaults to None.

      Returns:
         (Tensor, Tensor):
            - The first element is the order of each class.
            - The second element is the weight for each class for that order.
                This can be used as ``weight`` in ``nn.CrossEntropyLoss()``.

      Examples:
          >>> a = [0,1,1,1,1,4,4,2]
          >>> # unique -> (tensor([0, 1, 2, 4]), tensor([1, 4, 1, 2]))
          >>> print(get_class_weight(a, 'sklearn'))
          >>> print(get_class_weight(a, 'inverse_size'))
          >>> print(get_class_weight(a, 'inverse_sqrt_size'))
          >>> print(get_class_weight(a, 'inverse_proba'))
          >>> print(get_class_weight(a, 'inverse_sqrt_proba'))
          >>> print(get_class_weight(a, 'inverse_softmax'))
          >>> print(get_class_weight(a, 'inverse_sqrt_softmax'))
          >>> print(get_class_weight(a, total_nu_class=6))
          >>> # Output:
          >>> sklearn -> (tensor([0, 1, 2, 4]), tensor([2.0000, 0.5000, 2.0000, 1.0000]))
          >>> inverse_size -> (tensor([0, 1, 2, 4]), tensor([1.0000, 0.2500, 1.0000, 0.5000]))
          >>> inverse_sqrt_size -> (tensor([0, 1, 2, 4]), tensor([1.0000, 0.5000, 1.0000, 0.7071]))
          >>> inverse_proba -> (tensor([0, 1, 2, 4]), tensor([8., 2., 8., 4.]))
          >>> inverse_sqrt_proba -> (tensor([0, 1, 2, 4]), tensor([2.8284, 0.7071, 2.8284, 1.4142]))
          >>> inverse_softmax -> (tensor([0, 1, 2, 4]), tensor([24.8038,  1.2349, 24.8038,  9.1248]))
          >>> inverse_sqrt_softmax -> (tensor([0, 1, 2, 4]), tensor([4.9803, 1.1113, 4.9803, 3.0207]))
          >>> (tensor([0, 1, 2, 3, 4, 5]), tensor([2.0000, 0.5000, 2.0000, 0.0000, 1.0000, 0.0000]))

      """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    unique, counts = a.unique(return_counts=True)
    if method == 'sklearn':
        weights = len(a) / (len(a.unique()) * torch.bincount(a))
    elif method == 'inverse_size':
        weights = 1 / counts
    elif method == 'inverse_sqrt_size':
        weights = 1 / counts.sqrt()
    elif method == 'inverse_proba':
        weights = 1 / (counts / counts.sum())
    elif method == 'inverse_sqrt_proba':
        weights = 1 / (counts / counts.sum()).sqrt()
    elif method == 'inverse_softmax':
        weights = 1 / (torch.softmax(counts.float() / counts.max(), 0))
    elif method == 'inverse_sqrt_softmax':
        weights = 1 / (torch.softmax(counts.float() / counts.max(), 0)).sqrt()
    else:
        raise ValueError(f"method {method} is not supported")
    if total_nu_class is not None:
        full_weights = torch.zeros(total_nu_class, device=a.device)
        for i in range(len(unique)):
            full_weights[unique[i]] = weights[i]
        weights = full_weights
        unique = torch.arange(total_nu_class)
    return unique, weights


def model_keys(model):
    """Get the first-level layer name of the model.

      Args:
          model (nn.Module): PyTorch model.

      Returns:
          list: The keys of the model in order.

      Examples:
          >>> keys = model_keys(model)
          >>> for i in keys:
          >>>     getattr(model, i)
      """
    out = []
    for k, v in model.named_children():
        out.append(k)
    return out

def tensor_to_Dataset(x, y):
    if isinstance(x, torch.Tensor):  # Handle if x is tensor
        assert y is not None, f"x is {type(x)}, expected y to be torch.Tensor or List[Tensor]"
        if isinstance(y, (tuple, list)):  # Handle if y is list
            x = TensorDataset(x, *y)
        else:  # Handle if y is tensor
            x = TensorDataset(x, y)
    elif isinstance(x, (tuple, list)):  # Handle if x is list
        assert y is not None, f"x is {type(x)}, expected y to be torch.Tensor or List[Tensor]"
        if isinstance(y, (tuple, list)):  # Handle if y is list
            x = TensorDataset(*x, *y)
        else:  # Handle if y is tensor
            x = TensorDataset(*x, y)
    return x