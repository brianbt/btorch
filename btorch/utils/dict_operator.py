import operator


def dict_operator(ls, op):
    """operator on dicts.

    return ls[0] op ls[1] op ls[2] ... (in this order)
    
    Args:
        ls (List[Dict]): list of dicts.
          The first dict is the base dict.
          Expected the first dict contains all the keys in the other dicts.
        op (str or Callable): supported operators are '+', '-', '*', '/', '//', '@'.
          Callable should be ``operator.xxx`` or a function that take in two inputs and return one.

    Returns:
        dict: result of the operation.
    
    Examples:
        >>> a = {'a': torch.ones(4)+3, 'b': 5, 'c':7}
        >>> b = {'a': torch.ones(4), 'b': torch.randn(3)}
        >>> dict_operator([a, b], '+') 
        >>> # {'a': tensor([5., 5., 5., 5.]), 'b': tensor([6.4238, 4.6137, 4.4428]), 'c': 7}
        >>> a = {'a':torch.randn(1,2),'b':torch.randn(1,3)}
        >>> b = {'a':torch.randn(2,2)}
        >>> dict_operator([a,b], lambda x, y: torch.vstack([x,y]))
        >>> #{'a': tensor([[ 1.0247, -1.2278],
        >>> #              [ 0.4615, -1.5306],
        >>> #              [ 0.7885, -0.3302]]), 
        >>> # 'b': tensor([[0.1311, 0.9321, 0.6580]])}

    Note:
        If the there is a key in the other dicts that is not in the first dicts,
        it will add that key to the first dict and init the value as 0.
    """
    mapping = {'+': operator.add,
               '-': operator.sub,
               '*': operator.mul,
               '/': operator.truediv,
               '//': operator.floordiv,
               '@': operator.matmul}
    if isinstance(op, str):
        if op not in mapping:
            raise ValueError(f'not support `{op}`')
        op = mapping[op]
    out = dict()
    all_keys = set()
    for d in ls:
       all_keys.update(d.keys())
    for k in all_keys:
        for d in range(len(ls)):
            if d == 0 and k in ls[d]:
                out[k] = ls[d][k]
            elif d == 0 and k not in ls[d]:
                out[k] = 0
            else:
                if k in ls[d]:
                    out[k] = op(out[k], ls[d][k])
    return out

def dict_sum(ls):
    """see dict_operator()"""
    return dict_operator(ls, '+')

def dict_mul(ls):
    """see dict_operator()"""
    return dict_operator(ls, '*')

def dict_multiply(ls):
    """Alias for dict_mul()."""
    return dict_mul(ls)

def dict_sub(ls):
    """see dict_operator()"""
    return dict_operator(ls, '-')

def dict_subtract(ls):
    """Alias for dict_sub()."""
    return dict_subtract(ls)

def dict_div(ls):
    """see dict_operator()"""
    return dict_operator(ls, '/')

def dict_divide(ls):
    """Alias for dict_div()."""
    return dict_divide(ls)

def dict_combine(ls):
    """Turn `list of dict` to `dict of list`."""
    tmp = {}
    for dd in ls:
        for item in dd.keys():
            if item not in tmp:
                tmp[item] = []
            tmp[item].append(dd[item])
    return tmp