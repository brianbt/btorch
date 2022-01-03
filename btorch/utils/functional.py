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

def pad_to(x, shape, mode='constant', value=0.0):
    """pad a Tensor to desire shape, pad bottom/right first if shape is odd. 

    Args:
        x (Tensor): pytorch Tensor (*, H, W)
        shape (Tuple(int)): desired (Hd, Wd)
        mode: see `torch.nn.functional.pad`
        value: see `torch.nn.functional.pad`

    Returns:
        Tensor: pytorch Tensor (*, Hd, Wd)
    """
    to_pad = []
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

def test():
    print("testing")