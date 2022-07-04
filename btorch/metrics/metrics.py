import torch

def mini_batch_combiner(cached):
    """for mrmse mini-batch combine

    Args:
        cached (Tensor or List[Tensor]): (B, C) or list([C])

    Returns:
        int: metric loss

    Examples:
        >>> pre_batch_loss = []
        >>> for batch in num_batch:
        >>>     mse_b, c_b =  mrmse(y_pred_b,y_b)
        >>>     pre_batch_loss.append(c_b)
        >>> mse = mini_batch_combiner(pre_batch_loss) #CORRECT
        >>> mse = pre_batch_loss.mean() #WRONG
    """
    if isinstance(cached, list):
        cached = torch.cat(cached, 0)
    if cached.dim() != 2:
        raise ValueError("`cached` should have 2 dimension. (B, C)")
    mrmse = cached.nanmean(0)
    mrmse = torch.sqrt(mrmse)
    mrmse = torch.nanmean(mrmse)
    return mrmse


def all_rmse(count_pred, count_gt):
    """helper function for cacluating mrmse, mrmse_nz, rel_mrmse, rel_mrmse_nz
    """
    out = {
        'mrmse': mrmse(count_pred, count_gt),
        'mrmse_nz': mrmse(count_pred, count_gt, True),
        'rel_mrmse': rel_mrmse(count_pred, count_gt),
        'rel_mrmse_nz': rel_mrmse(count_pred, count_gt, True),
    }
    return out


def mrmse(count_pred, count_gt, non_zero=False):
    """compute mean Root Mean Square Error

        RMSE = (y_pred - y)**2.mean(axis=0).sqrt()
          RMSE for each class (C)
        mRMSE = RMSE.mean()
          mean of RMSE across class (C)

    Args:
        count_pred (Tensor): (N,C) tensor prediction
        count_gt (Tensor): (N,C) tensor ground truth
        non_zero (bool): calc metric for non-zero elements only if True

    Returns:
        int: metric score
        Tensor: (1,C) cache for mini-batch caluation
    """
    ## compute mrmse
    nzero_mask = torch.ones(count_gt.size(), device=count_pred.device)
    if non_zero:
        nzero_mask = torch.zeros(count_gt.size(), device=count_pred.device)
        nzero_mask[count_gt != 0] = 1
    mrmse = torch.pow(count_pred - count_gt, 2)
    # print(mrmse)
    mrmse = torch.mul(mrmse, nzero_mask)
    # print(mrmse)
    mrmse = torch.sum(mrmse, 0)
    # print(mrmse)
    nzero = torch.sum(nzero_mask, 0)
    mrmse = torch.div(mrmse, nzero)
    # print(mrmse)
    cache = mrmse
    mrmse = torch.sqrt(mrmse)
    # print(mrmse)
    mrmse = torch.nanmean(mrmse)
    return mrmse, cache.unsqueeze(0)


def rel_mrmse(count_pred, count_gt, non_zero=False):
    """compute rel mean Root Mean Square Error

        RMSE = (y_pred - y)**2.mean(axis=0).sqrt()
          RMSE for each class (C)
        rel_RMSE = RMSE/(y+1)
        mrel_RMSE = rel_RMSE.mean()
          mean of rel_RMSE across class (C)

    Args:
        count_pred (Tensor): (N,C) tensor prediction
        count_gt (Tensor): (N,C) tensor ground truth
        non_zero (bool): calc metric for non-zero elements only if True

    Returns:
        int: metric score
        Tensor: (1,C) cache for mini-batch caluation
    """
    ## compute relative mrmse
    nzero_mask = torch.ones(count_gt.size(), device=count_pred.device)
    if non_zero:
        nzero_mask = torch.zeros(count_gt.size(), device=count_pred.device)
        nzero_mask[count_gt != 0] = 1
    num = torch.pow(count_pred - count_gt, 2)
    denom = count_gt.clone()+1
    rel_mrmse = torch.div(num, denom)
    rel_mrmse = torch.mul(rel_mrmse, nzero_mask)
    rel_mrmse = torch.sum(rel_mrmse, 0)
    nzero = torch.sum(nzero_mask, 0)
    rel_mrmse = torch.div(rel_mrmse, nzero)
    cache = rel_mrmse
    rel_mrmse = torch.sqrt(rel_mrmse)
    rel_mrmse = torch.nanmean(rel_mrmse)
    return rel_mrmse, cache.unsqueeze(0)


def accuarcy(model_output, y_true, reduction='sum', method='multiclass'):
    """scoring function, can be directly used in ``test_epoch()``
    
    Args:
        model_output (Tensor): should be ``(N,C)`` or ``(N)``
        y_true (_type_): should be ``(N)``
        reduction (str, optional): either ``sum`` or ``mean``. Defaults to 'sum'.
        method (str, optional): either ``multiclass`` or ``binary``. Defaults to 'multiclass'.
          If your loss is CrossEntropyLoss, you should use ``multiclass``.
          If your loss is BSELoss, you should use ``binary``.

    Returns:
        If reduction is ``none``, same shape as the target. Otherwise, scalar.
    """
    if method == 'multiclass':
        y_pred = model_output.max(1)[1]
    elif method == 'binary':
        y_pred = (model_output>=0).int().float()
    else:
        raise ValueError("method should be 'multiclass' or 'binary'")
    # print(y_pred)
    if reduction == 'sum':
        out = (y_pred == y_true).float().sum().item()
    elif reduction == 'mean':
        out = (y_pred == y_true).float().sum().item()
    elif reduction is None:
        out = (y_pred.view(-1) == y_true.view(-1)).float()
    else:
         raise ValueError("reduction should be 'mean' or 'sum'")
    return out
