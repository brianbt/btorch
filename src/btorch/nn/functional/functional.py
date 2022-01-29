import torch
import torch.nn.functional as F

def adaptive_pad1d(x, shape, mode='constant', value=0.0):
    """pad a 1D Tensor to desire shape, pad right first if shape is odd. 

    Args:
        x (Tensor): pytorch Tensor (*, D)
        shape (List(int)): desired (*, Dd). Only the last value will be used
        mode: see `torch.nn.functional.pad`
        value: see `torch.nn.functional.pad`

    Returns:
        Tensor: pytorch Tensor (*, Dd)
    """
    to_pad = []
    if not isinstance(shape, int):
        shape = shape[-1]
    new_s = shape - x.size(-1)
    if new_s % 2 == 0:
        to_pad.append(new_s // 2)
        to_pad.append(new_s // 2)
    else:
        to_pad.append(new_s // 2)
        to_pad.append(new_s // 2 + 1)
    out = F.pad(x, to_pad, mode, value)
    return out

def adaptive_pad2d(x, shape, mode='constant', value=0.0):
    """pad a 2D Tensor to desire shape, pad bottom/right first if shape is odd. 

    Args:
        x (Tensor): pytorch Tensor (*, H, W)
        shape (List(int)): desired (*, Hd, Wd). Only the last two value will be used
        mode: see `torch.nn.functional.pad`
        value: see `torch.nn.functional.pad`

    Returns:
        Tensor: pytorch Tensor (*, Hd, Wd)
    """
    to_pad = []
    shape = shape[-2:]
    new_w = shape[1] - x.size(-1)
    new_h = shape[0] - x.size(-2)
    if new_w % 2 == 0:
        to_pad.append(new_w // 2)
        to_pad.append(new_w // 2)
    else:
        to_pad.append(new_w // 2)
        to_pad.append(new_w // 2 + 1)
    if new_h % 2 == 0:
        to_pad.append(new_h // 2)
        to_pad.append(new_h // 2)
    else:
        to_pad.append(new_h // 2)
        to_pad.append(new_h // 2 + 1)
    out = F.pad(x, to_pad, mode, value)
    return out

def adaptive_pad3d(x, shape, mode='constant', value=0.0):
    """pad a 3D Tensor to desire shape 

    Args:
        x (Tensor): pytorch Tensor (*, D1, D2, D3)
        shape (List(int)): desired (*, D1d, D2d, D3d). Only the last three value will be used
        mode: see `torch.nn.functional.pad`
        value: see `torch.nn.functional.pad`

    Returns:
        Tensor: pytorch Tensor (*, D1d, D2d, D3d)
    """
    to_pad = []
    shape = shape[-3:]
    new_d1 = shape[2] - x.size(-1)
    new_d2 = shape[1] - x.size(-2)
    new_d3 = shape[0] - x.size(-3)
    if new_d1 % 2 == 0:
        to_pad.append(new_d1 // 2)
        to_pad.append(new_d1 // 2)
    else:
        to_pad.append(new_d1 // 2)
        to_pad.append(new_d1 // 2 + 1)
    if new_d2 % 2 == 0:
        to_pad.append(new_d2 // 2)
        to_pad.append(new_d2 // 2)
    else:
        to_pad.append(new_d2 // 2)
        to_pad.append(new_d2 // 2 + 1)
    if new_d3 % 2 == 0:
        to_pad.append(new_d3 // 2)
        to_pad.append(new_d3 // 2)
    else:
        to_pad.append(new_d3 // 2)
        to_pad.append(new_d3 // 2 + 1)
    out = F.pad(x, to_pad, mode, value)
    return out


def multi_hot(ls, num_classes=-1):
    """Take in List and turn it to multi-hot encoding
    
    See https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html for details
    
    Args:
        ls (List[int or List[int]]): List contains the class(es) of that element
        num_classes (int, optional): Total number of classes.
    
    Returns:
        LongTensor: multi-hot encoding

    Notice: 
      This function can be used to replace F.one_hot()
      This may be slow if there are many element has more than one class

    Examples:
        >>> # input can be either in these three format
        >>> arr = [1,2,3,4,5]
        >>> arr = [1,2,[9,2],3]
        >>> arr = [[1,2,3], [2,3], [4,5], [9,2]]
        >>> multi_hot(arr)
    """
    if isinstance(ls, torch.Tensor):
        return F.one_hot(ls, num_classes=num_classes)
    try:
        return F.one_hot(torch.tensor(ls), num_classes=num_classes)
    except:
        pass
    to_fix = []
    to_torch = []
    for w in range(len(ls)):
        if isinstance(ls[w], int):
            to_torch.append(ls[w])
        else:
            to_torch.append(max(ls[w]))
            to_fix.append(w)
    label = F.one_hot(torch.tensor(to_torch), num_classes=num_classes)
    for fix in to_fix:
        for w in ls[fix]:
            label[fix][w] = 1
    return label