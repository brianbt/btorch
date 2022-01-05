import torch
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        """ Undo Transforms.Normalize()

        Args:
            mean (iterable or Tensor): Should be same as Transforms.Normalize args
            std (iterable or Tensor): Should be same as Transforms.Normalize args

        Examples:
        >>> mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        >>> transform = transforms.Compose([transforms.Normalize(mean, std)])
        >>> unNormer = UnNormalize(mean, std)
        >>> # transformed_input is from dataloader(transformed_dataset)
        >>> plt.imshow(unNormer(transformed_input))
        """
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Accept (N,C,H,W), (N,H,W,C), (C,H,W) and (H,W,C)
        Returns:
            Tensor: Normalized image (N,H,W,C). Will be on CPU
        """
        # Turn CHW to HWC
        tensor = tensor.to('cpu')
        if len(tensor.shape) == 3 and tensor.shape[2] != 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        elif len(tensor.shape) == 4 and tensor.shape[3] != 3 and tensor.shape[1] == 3:
            tensor = tensor.permute(0, 2, 3, 1)
        tensor = ((tensor * self.std) + self.mean)
        return tensor

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, model=None):
    """
    Utility function for computing output of convolutions. This function did not calculate out_channels 
    
    Args:
        h_w (Tuple[] or Tensor): (*,H,W) -> Must be at least two dimension, OR, int when H=W.
        model (nn.Conv2d): get the parameter from the given model. Override all other parameters.
    """
    if model is not None:
        kernel_size = model.kernel_size
        stride = model.stride
        pad = model.padding
        dilation = model.dilation
        if isinstance(pad, str):
            raise ValueError(f"model.pad is str ({pad}). Only support int.")
    out = []
    if isinstance(h_w, torch.Tensor):
        out += list(h_w.shape[:-2])
        h_w = h_w.shape[-2:]
    elif isinstance(h_w, tuple) or isinstance(h_w, list):
        out += h_w[:-2]
        h_w = h_w[-2:]
    elif not isinstance(h_w, tuple) and not isinstance(h_w, list):
        h_w = (h_w, h_w)


    if not isinstance(kernel_size, tuple) and not isinstance(kernel_size, list):
        kernel_size = (kernel_size, kernel_size)

    if not isinstance(stride, tuple) and not isinstance(stride, list):
        stride = (stride, stride)

    if not isinstance(pad, tuple) and not isinstance(pad, list):
        pad = (pad, pad)

    if not isinstance(dilation, tuple) and not isinstance(dilation, list):
        dilation = (dilation, dilation)

    # print(h_w, kernel_size, stride, pad, dilation)
    h = (h_w[0] + (2 * pad[0]) - dilation[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - dilation[0]*(kernel_size[1]-1) - 1) // stride[1] + 1
    out.append(h)
    out.append(w)
    return tuple(out)

def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, model=None):
    """
    Utility function for computing output of transposed convolutions. This function did not calculate out_channels 
    
    Args:
        h_w (Tuple[] or Tensor): (*,H,W) -> Must be at least two dimension, OR, int when H=W.
        model (nn.ConvTranspose2d): get the parameter from the given model. Override all other parameters.
    """

    if model is not None:
        kernel_size = model.kernel_size
        stride = model.stride
        pad = model.padding
        dilation = model.dilation
        if isinstance(pad, str):
            raise ValueError(f"model.pad is str ({pad}). Only support int.")
    out = []
    if isinstance(h_w, torch.Tensor):
        out += list(h_w.shape[:-2])
        h_w = h_w.shape[-2:]
    elif isinstance(h_w, tuple) or isinstance(h_w, list):
        out += h_w[:-2]
        h_w = h_w[-2:]
    elif not isinstance(h_w, tuple) and not isinstance(h_w, list):
        h_w = (h_w, h_w)

    if not isinstance(kernel_size, tuple) and not isinstance(
            kernel_size, list):
        kernel_size = (kernel_size, kernel_size)

    if not isinstance(stride, tuple) and not isinstance(stride, list):
        stride = (stride, stride)

    if not isinstance(pad, tuple) and not isinstance(pad, list):
        pad = (pad, pad)

    if not isinstance(dilation, tuple) and not isinstance(dilation, list):
        dilation = (dilation, dilation)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + dilation[0]*(kernel_size[0]-1) + pad[0] + 1
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + dilation[1]*(kernel_size[1]-1) + pad[1] + 1
    out.append(h)
    out.append(w)
    return tuple(out)

def high_pass_filter(img, method='3'):
    """apply high-pass filter on image

    Args:
        img (np.ndarray): (C,H,W)
        method (str, optional): either `gaussian`, `3` or `5`. Defaults to '3'.

    Returns:
        np.ndarray: (C,H,W)
    """
    from scipy import ndimage
    # Gaussian high pass
    if method.lower() == 'gaussian':
        lowpass = ndimage.gaussian_filter(img, 3)
        highpass = img - lowpass
        highpass[highpass <= 0.03] = 0
        highpass[highpass > 0.03] = 1
        return highpass
    if method == '3':
        # 3x3 high pass
        kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])
    elif method == '5':
        # 5X5 high pass
        kernel = np.array([[-1, -1, -1, -1, -1], [-1, 1, 2, 1, -1], [-1, 2, 4, 2, -1],
                        [-1, 1, 2, 1, -1], [-1, -1, -1, -1, -1]])
    elif method == 'fun':
        kernel = np.array([[-1, -1, -1, -1, -1, -1, -1],
                        [-1,  1,  2,  3,  2,  1, -1],
                        [-1,  2,  3,  4,  3,  2, -1],
                        [-1,  3,  4,  5,  4,  3, -1],
                        [-1,  2,  3,  4,  3,  2, -1],
                        [-1,  1,  2,  3,  2,  1, -1],
                        [-1, -1, -1, -1, -1, -1, -1]])
    else:
        raise ValueError(f"Not support method ({method})")
    kernel = kernel[:, :, None]
    highpass = ndimage.convolve(img, kernel)
    highpass[highpass <= 0.03] = 0
    highpass[highpass > 0.03] = 1
    return highpass