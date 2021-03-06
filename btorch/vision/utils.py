import math
import warnings

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


class UnNormalize(object):
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
    def __init__(self, mean, std):
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


def conv_output_shape(input_size, kernel_size=1, stride=1, pad=0, dilation=1, model=None):
    """
    Utility function for computing output of convolutions. This function did not calculate out_channels 
    
    Args:
        input_size (Tuple[] or Tensor): Input Tensor shape. (*,H,W) -> Must be at least two dimension, OR, int when H=W.
        model (nn.Conv2d): get the parameter from the given model. Override all other parameters.
    """
    out_channels = None
    if model is not None:
        kernel_size = model.kernel_size
        stride = model.stride
        pad = model.padding
        dilation = model.dilation
        out_channels = model.out_channels
        if isinstance(pad, str):
            raise ValueError(f"model.pad is str ({pad}). Only support int.")
    out = []
    if isinstance(input_size, torch.Tensor):
        out += list(input_size.shape[:-2])
        if out_channels is not None:
            out[-1] = out_channels
        input_size = input_size.shape[-2:]
    elif isinstance(input_size, tuple) or isinstance(input_size, list):
        out += input_size[:-2]
        if out_channels is not None:
            out[-1] = out_channels
        input_size = input_size[-2:]
    elif not isinstance(input_size, tuple) and not isinstance(input_size, list):
        input_size = (input_size, input_size)

    if not isinstance(kernel_size, tuple) and not isinstance(kernel_size, list):
        kernel_size = (kernel_size, kernel_size)

    if not isinstance(stride, tuple) and not isinstance(stride, list):
        stride = (stride, stride)

    if not isinstance(pad, tuple) and not isinstance(pad, list):
        pad = (pad, pad)

    if not isinstance(dilation, tuple) and not isinstance(dilation, list):
        dilation = (dilation, dilation)

    h = (input_size[0] + (2 * pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    w = (input_size[1] + (2 * pad[1]) - dilation[0] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    out.append(h)
    out.append(w)
    return tuple(out)


def convtransp_output_shape(input_size, kernel_size=1, stride=1, pad=0, dilation=1, model=None):
    """
    Utility function for computing output of transposed convolutions. This function did not calculate out_channels 
    
    Args:
        input_size (Tuple[] or Tensor): Input Tensor shape. (*,H,W) -> Must be at least two dimension, OR, int when H=W.
        model (nn.ConvTranspose2d): get the parameter from the given model. Override all other parameters.
    """
    out_channels = None
    if model is not None:
        kernel_size = model.kernel_size
        stride = model.stride
        pad = model.padding
        dilation = model.dilation
        out_channels = model.out_channels
        if isinstance(pad, str):
            raise ValueError(f"model.pad is str ({pad}). Only support int.")
    out = []
    if isinstance(input_size, torch.Tensor):
        out += list(input_size.shape[:-2])
        if out_channels is not None:
            out[-1] = out_channels
        input_size = input_size.shape[-2:]
    elif isinstance(input_size, tuple) or isinstance(input_size, list):
        out += input_size[:-2]
        if out_channels is not None:
            out[-1] = out_channels
        input_size = input_size[-2:]
    elif not isinstance(input_size, tuple) and not isinstance(input_size, list):
        input_size = (input_size, input_size)

    if not isinstance(kernel_size, tuple) and not isinstance(
            kernel_size, list):
        kernel_size = (kernel_size, kernel_size)

    if not isinstance(stride, tuple) and not isinstance(stride, list):
        stride = (stride, stride)

    if not isinstance(pad, tuple) and not isinstance(pad, list):
        pad = (pad, pad)

    if not isinstance(dilation, tuple) and not isinstance(dilation, list):
        dilation = (dilation, dilation)

    h = (input_size[0] - 1) * stride[0] - 2 * pad[0] + dilation[0] * (kernel_size[0] - 1) + pad[0] + 1
    w = (input_size[1] - 1) * stride[1] - 2 * pad[1] + dilation[1] * (kernel_size[1] - 1) + pad[1] + 1
    out.append(h)
    out.append(w)
    return tuple(out)


def conv_kernel_shape(input_size, output_size, stride=1, pad=0, dilation=1):
    """
    Utility function for computing kernel size of convolutions using the expected output size. This function ignores channels 
    
    Args:
        input_size (Tuple[] or Tensor): Input Tensor shape. (*,H,W) -> Must be at least two dimension, OR, int when H=W.
        output_size (Tuple[] or Tensor): Expected output shape. (*,H,W) -> Must be at least two dimension, OR, int when H=W.
    """
    out = []
    if isinstance(input_size, torch.Tensor):
        out += list(input_size.shape[:-2])
        input_size = input_size.shape[-2:]
    elif isinstance(input_size, tuple) or isinstance(input_size, list):
        out += input_size[:-2]
        input_size = input_size[-2:]
    elif not isinstance(input_size, tuple) and not isinstance(input_size, list):
        input_size = (input_size, input_size)

    if isinstance(output_size, torch.Tensor):
        output_size = list(output_size.shape[-2:])
    elif isinstance(output_size, tuple) or isinstance(output_size, list):
        output_size = output_size[-2:]
    elif not isinstance(output_size, tuple) and not isinstance(output_size, list):
        output_size = (output_size, output_size)

    if not isinstance(stride, tuple) and not isinstance(stride, list):
        stride = (stride, stride)

    if not isinstance(pad, tuple) and not isinstance(pad, list):
        pad = (pad, pad)

    if not isinstance(dilation, tuple) and not isinstance(dilation, list):
        dilation = (dilation, dilation)

    k_h = (output_size[0] * stride[0] - stride[0] - input_size[0] - 2 * pad[0] + 1) // -dilation[0] + 1
    k_w = (output_size[1] * stride[1] - stride[1] - input_size[1] - 2 * pad[1] + 1) // -dilation[1] + 1
    out.append(k_h)
    out.append(k_w)
    return tuple(out)


def convtransp_kernel_shape(input_size, output_size, stride=1, pad=0, dilation=1):
    """
    Utility function for computing kernel size of transposed convolutions using the expected output size. This function ignores channels 
    
    Args:
        input_size (Tuple[] or Tensor): Input Tensor shape. (*,H,W) -> Must be at least two dimension, OR, int when H=W.
        output_size (Tuple[] or Tensor): Expected output shape. (*,H,W) -> Must be at least two dimension, OR, int when H=W.
    """
    out = []
    if isinstance(input_size, torch.Tensor):
        out += list(input_size.shape[:-2])
        input_size = input_size.shape[-2:]
    elif isinstance(input_size, tuple) or isinstance(input_size, list):
        out += input_size[:-2]
        input_size = input_size[-2:]
    elif not isinstance(input_size, tuple) and not isinstance(input_size, list):
        input_size = (input_size, input_size)

    if isinstance(output_size, torch.Tensor):
        output_size = list(output_size.shape[-2:])
    elif isinstance(output_size, tuple) or isinstance(output_size, list):
        output_size = output_size[-2:]
    elif not isinstance(output_size, tuple) and not isinstance(output_size, list):
        output_size = (output_size, output_size)

    if not isinstance(stride, tuple) and not isinstance(stride, list):
        stride = (stride, stride)

    if not isinstance(pad, tuple) and not isinstance(pad, list):
        pad = (pad, pad)

    if not isinstance(dilation, tuple) and not isinstance(dilation, list):
        dilation = (dilation, dilation)

    k_h = (output_size[0] - stride[0] * (input_size[0] - 1) + pad[0] - 1) // dilation[0] + 1
    k_w = (output_size[1] - stride[1] * (input_size[1] - 1) + pad[1] - 1) // dilation[1] + 1
    out.append(k_h)
    out.append(k_w)
    return tuple(out)


def high_pass_filter(img, method='3'):
    """apply high-pass filter on image

    Args:
        img (np.ndarray): (C,H,W)
        method (str, optional): either ``gaussian``, ``3`` or ``5``. Defaults to '3'.

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
                           [-1, 8, -1],
                           [-1, -1, -1]])
    elif method == '5':
        # 5X5 high pass
        kernel = np.array([[-1, -1, -1, -1, -1], [-1, 1, 2, 1, -1], [-1, 2, 4, 2, -1],
                           [-1, 1, 2, 1, -1], [-1, -1, -1, -1, -1]])
    elif method == 'fun':
        kernel = np.array([[-1, -1, -1, -1, -1, -1, -1],
                           [-1, 1, 2, 3, 2, 1, -1],
                           [-1, 2, 3, 4, 3, 2, -1],
                           [-1, 3, 4, 5, 4, 3, -1],
                           [-1, 2, 3, 4, 3, 2, -1],
                           [-1, 1, 2, 3, 2, 1, -1],
                           [-1, -1, -1, -1, -1, -1, -1]])
    else:
        raise ValueError(f"Not support method ({method})")
    kernel = kernel[:, :, None]
    highpass = ndimage.convolve(img, kernel)
    highpass[highpass <= 0.03] = 0
    highpass[highpass > 0.03] = 1
    return highpass


def find_corner_from_mask(A, flexible=False):
    """Extract a rectangular bounding box of each object color
    
    Each pixel value in A will be treated as one object.
    In other words, each color is treated as one object.
    Expected A to be a very simple image that contains only few colors.
    Usually is a mask of the object.
    
    Args:
        A (np.array): can be 2D(grayscale) or 3D(RGB)
          if input is 3D(RGB): will converge to 2D
        flexible (float): percent of enlargement on the topLeft and bottomRight.
    Returns:
        np.array: (2D)grayscale version of input A.
        dict: key is object index, value is [x_min, y_min, x_max, y_max] coordinates
    """
    import math
    if not isinstance(A, np.ndarray):
        A = np.asarray(A)
    if len(A.shape) == 3 and A.shape[0] == 3:
        # CHW -> HWC, C=3
        A = A.transpose(1, 2, 0)
    if len(A.shape) == 3:
        # turn to grayscale
        A = np.dot(A[..., :3], [0.299, 0.587, 0.114])
    toFind = np.unique(A)
    out = dict()
    for tar in toFind:
        if tar == 0:
            continue
        i, j = np.where(A == tar)
        topLeft = (min(i), min(j))
        bottomRight = (max(i), max(j))
        if flexible:
            topLeft = (math.floor(topLeft[0] * (1 - flexible)), math.floor(topLeft[1] * (1 - flexible)))
            bottomRight = (math.floor(bottomRight[0] * (1 + flexible)), math.floor(bottomRight[1] * (1 + flexible)))
        out[tar] = [topLeft, bottomRight]
    return A, out


def replace_part(img, part, loc, handle_transparent=False):
    """replace part of ``img`` by ``part``. It supports batch mode.

    Args:
        img (Tensor or ndarray): either (N,C,H,W) or (C,H,W)
        part (Tensor or ndarray): used to replace. (C,H,W)
        loc (List[int]): [x_min, x_max], the topLeft coordinate of img to be replaced.
        handle_transparent (bool): if the pixel that has zero value (transparent) will be replaced or not.

    Returns:
        Tensor: The replaced version of img

    Examples:
        >>> img = torch.zeros((2,3,10,10))
        >>> part = torch.ones((3,4,2))
        >>> replace_part(img, a, [4,1])
    """
    img = img.clone()
    x, y = loc
    part_mask = (part.sum(0) > 0).expand_as(part)
    if len(img.shape) == 4:
        if handle_transparent:
            print(part_mask.shape)
            part_mask = part_mask.expand([img.shape[0]] + list(part.shape))
            part = part.expand([img.shape[0]] + list(part.shape))
            print(img.shape, img[:, :, x:x + part.shape[-2], y:y + part.shape[-1]].shape, part.shape, part_mask.shape)
            print(img[:, :, x:x + part.shape[-2], y:y + part.shape[-1]][part_mask].shape)
            print(part[part_mask].shape)
            img[:, :, x:x + part.shape[-2], y:y + part.shape[-1]][part_mask] = part[part_mask]
        else:
            img[:, :, x:x + part.shape[-2], y:y + part.shape[-1]] = part
    elif len(img.shape) == 3:
        if handle_transparent:
            img[:, x:x + part.shape[-2], y:y + part.shape[-1]][part_mask] = part[part_mask]
        else:
            img[:, x:x + part.shape[-2], y:y + part.shape[-1]] = part
    else:
        raise ValueError(f"img's dim should be either 3 or 4, got {len(img.shape)}, {img.shape}")
    return img


def crop_img_from_bbx(img, bboxs, bbox_format='pascal', raw=False, return_dict=False):
    """Crop the bounding box of an image from bbox

    Args:
        img (Tensor): (3,H,W). NOTICE: this img should be in original size. Without any Resize()
        bboxs (bounding boxes)
          it accepts:
            - (List[List]): List of bbox if raw is False
            - (List[Dict[]]): List of dictionary if raw is True.
                This should be the default target output of ``datasets.COCODection()``
        bbox_format (str, optional): Format of bboxs.
          Either ``pascal`` or ``coco``. Defaults to 'pascal'.
        raw (bool, optional): see bboxs doc. Defaults to False.
        return_dict (bool, optional): only support when ``raw`` is ON

    Returns:
        List[Tensor]: List of Cropped Images
        Dict[category_id:[Tensor]]: Dict of List of Cropped Images for each class

    Example:
        >>> for img, target in datasets.COCODection():
        >>>     break
        >>> datasets.crop_img_from_bbx(img, target, bbox_format='coco', raw=True, return_dict=True)
    """
    if img.shape[1] == img.shape[2]:
        warnings.warn("Did you resized this image? If yes -> cropping will NOT be correct")
    if return_dict and not raw:
        raise ValueError("return_dict=True only support when raw is True")

    if bbox_format.lower() == 'pascal':
        dict_check = get_pascal_class_dict(reverse=True)
    if raw:
        tmp = []
        classes = []
        if bbox_format.lower() == 'coco':
            for i in bboxs:
                tmp.append(i['bbox'])
                classes.append(i['category_id'])
        else:
            if not isinstance(bboxs, dict):
                raise Exception("``bboxs`` should be a Dict")
            for i in bboxs['annotation']['object']:
                tmp.append(list(i['bndbox'].values()))
                classes.append(dict_check[i['name']])
        bboxs = tmp
    if not isinstance(bboxs[0], list):
        bboxs = [bboxs]
    out = {} if return_dict else []
    for idx in range(len(bboxs)):
        bbox = [int(x) for x in bboxs[idx]]
        X, Y, H, W = bbox
        if bbox_format.lower() == 'coco':
            cropped_image = img[:, Y:Y + H, X:X + W]
        else:
            cropped_image = img[:, Y:W, X:H]
        # print(f'{img.shape} -> {cropped_image.shape}, {bbox}')
        if return_dict:
            if classes[idx] not in out:
                out[classes[idx]] = []
            out[classes[idx]].append(cropped_image)
        else:
            out.append(cropped_image)
    return out


def get_pascal_class_dict(reverse=False):
    """"Get Pascal VOC idx2label dict. reverse will give label2idx.
    """
    out = dict(
        zip(range(20), [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]))
    if reverse:
        return {v: k for k, v in out.items()}
    else:
        return out


def get_COCO_class_dict(include_background=False, reverse=False):
    """ Get COCO 2014 idx2label dict. reverse will give label2idx.
    """
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    if include_background:
        out = dict(
            zip(range(81), classes)
        )
    else:
        out = dict(
            zip(range(80), classes[1:])
        )
    if reverse:
        return {v: k for k, v in out.items()}
    else:
        return out


def get_COCO_paper_class_dict(include_background=False, reverse=False):
    """ Get COCO Paper idx2label dict. reverse will give label2idx.
    """
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
        'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush', 'hair brush'
    ]

    if include_background:
        out = dict(zip(range(92), classes))
    else:
        out = dict(zip(range(91), classes[1:]))
    if reverse:
        return {v: k for k, v in out.items()}
    else:
        return out


def coco_ann2Mask(img, annotations, return_dict=False):
    """Generate Mask for each object on COCO dataset

    Args:
        img (Tensor): (*,H,W)
        annotations (List[Dict[]]): This should be the default target output of datasets.COCODection()
    
    Returns:
        Dict[id:List[MASK]]: Each mask is a 2D matrix that contains either 0, 1. Size is same as ``img``
    """

    def decodeSeg(mask, segmentations):
        """
        Draw segmentation
        """
        pts = [
            np.array(anno).reshape(-1, 2).round().astype(int)
            for anno in segmentations
        ]
        mask = cv2.fillPoly(mask, pts, 1)

        return mask

    def decodeRl(mask, rle):
        """
        Run-length encoded object decode
        """
        mask = mask.reshape(-1, order='F')

        last = 0
        val = True
        for count in rle['counts']:
            val = not val
            mask[last:(last + count)] |= val
            last += count

        mask = mask.reshape(rle['size'], order='F')
        return mask

    h, w = img.shape[-2], img.shape[-1]
    out = {} if return_dict else []
    for annotation in annotations:
        mask = np.zeros((h, w), np.uint8)
        segmentations = annotation['segmentation']
        if isinstance(segmentations, list):  # segmentation
            mask = decodeSeg(mask, segmentations)
        else:  # run-length
            mask = decodeRl(mask, segmentations)
        if return_dict:
            if annotation['category_id'] not in out:
                out[annotation['category_id']] = []
            out[annotation['category_id']].append(mask)
        else:
            out.append(mask)
    return out


def smart_rotate(image):
    """Fix the rotation error when PIL.Image.open()

    Sometimes, when we open an image that is taken by our cell phone,
    The rotation maybe wrong, this function helps to fix that error.

    Args:
        image (PIL.Image): Python Image

    Returns:
        PIL.Image: Python Image
    """
    from PIL import ImageOps
    image = image
    exif = image.getexif()
    # Remove all exif tags
    for k in exif.keys():
        if k != 0x0112:
            exif[k] = None  # If I don't set it to None first (or print it) the del fails for some reason.
            del exif[k]
    # Put the new exif object in the original image
    new_exif = exif.tobytes()
    image.info["exif"] = new_exif
    # Rotate the image
    transposed = ImageOps.exif_transpose(image)
    return transposed


def img_MinMaxScaler(img, feature_range=(0, 1)):
    """MinMaxScaler for img data, it support 2D(gray), 3D and 4D(batch) data.
    
    sklearn.MinMaxScaler only works for 2D. 
    Usually used when visualizing the image data extracted from hidden layers.
    The visualized may look like ``UnNormalize()``, but the pixel value obtained by this function is NOT exactly the same the un-normalized one.
    To perform un-normalization, please use ``UnNormalize()`` instead.
    It can also be used to rescale the pixel value to the given range.

    Args:
        img (Tensor): pytorch Tensor. (H,W) or (C,H,W) or (N,C,H,W)
        feature_range (tuple, optional): Defaults to (0, 1).

    Returns:
        Tensor: scaled Image, same shape as input. dtype can be int or float depends on the feature_range.

    Note:
        >>> plt.imshow(img.permute(1,2,0))
        >>> # If have warning ``Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).``
        >>> # You can try to do the following, beware that the visualized output will look like UnNormalized.
        >>> plt.imshow(img_MinMaxScaler(img).permute(1,2,0))
    """
    min_v, max_v = feature_range
    if len(img.shape) == 2:
        X_std = (img - img.amin((-2, -1)).view(1).unsqueeze(-1).expand(img.shape)) / (
            (img.amax((-2, -1)) - img.amin((-2, -1))).view(1).unsqueeze(-1).expand(img.shape))
    elif len(img.shape) == 3:
        X_std = (img - img.amin((-2, -1)).view(3, 1).unsqueeze(-1).expand(img.shape)) / (
            (img.amax((-2, -1)) - img.amin((-2, -1))).view(3, 1).unsqueeze(-1).expand(img.shape))
    elif len(img.shape) == 4:
        X_std = (img - img.amin((-2, -1)).view(-1, 3, 1).unsqueeze(-1).expand(img.shape)) / (
            (img.amax((-2, -1)) - img.amin((-2, -1))).view(-1, 3, 1).unsqueeze(-1).expand(img.shape))
    else:
        raise ValueError("img must be either (H,W) or (C,H,W) or (N,C,H,W)")
    X_scaled = X_std * (max_v - min_v) + min_v
    if max_v == 1 and min_v == 0:
        return X_scaled
    else:
        return X_scaled.int()


def pplot(x, in_one_figure=True):
    """Smart plot for pytorch tensor.

    It handles every possible shape of tensor.

    Args:
        x (Tensor): accept (N,C,H,W), (N,H,W,C), (C,H,W), (H,W,C), (H,W).
          Only support ``gray`` and ``rgb`` image.
        in_one_figure (bool): if true, when x contains multi image, plot them in one figure.
        
            Note:
                Becareful when x has too many images, each plot might be very small.
    """

    def plot_3dim(x, return_array=False):
        if (x.shape[0] != 1 and x.shape[0] != 3) and (x.shape[-1] != 1 and x.shape[-1] != 3):
            raise ValueError('This is not a RGB or gray-scale image')
        if x.shape[0] == 3:
            if return_array:
                return (x.detach().cpu().permute(1, 2, 0), 'viridis')
            plt.imshow(x.detach().cpu().permute(1, 2, 0))
            plt.show()
        elif x.shape[0] == 1:
            if return_array:
                return (x.detach().cpu()[0], 'gray')
            plt.imshow(x.detach().cpu()[0], cmap='gray')
            plt.show()
        elif x.shape[-1] == 3:
            if return_array:
                return (x.detach().cpu(), 'viridis')
            plt.imshow(x.detach().cpu())
            plt.show()
        elif x.shape[-1] == 1:
            if return_array:
                return (x.detach().cpu()[:, :, 0], 'gray')
            plt.imshow(x.detach().cpu()[:, :, 0], cmap='gray')
            plt.show()

    if len(x.shape) == 4:
        if in_one_figure and x.shape[0] != 1:
            to_plot = []
            for i in range(x.shape[0]):
                to_plot.append(plot_3dim(x[i], return_array=True))
            fig = plt.figure(figsize=(8, 8))
            columns = math.ceil(math.sqrt(len(x)))
            rows = math.ceil(math.sqrt(len(x)))
            for i in range(1, len(to_plot) + 1):
                fig.add_subplot(rows, columns, i)
                plt.imshow(to_plot[i - 1][0], cmap=to_plot[i - 1][1])
            plt.show()
        else:
            for i in range(x.shape[0]):
                plot_3dim(x[i])
    elif len(x.shape) == 3:
        plot_3dim(x)
    elif len(x.shape) == 2:
        plt.imshow(x.detach().cpu(), cmap='gray')
        plt.show()
    else:
        raise ValueError('This is not a RGB or gray-scale image')


def flood_fill(img, target_value, fill_value, start_point, flags=4):
    """cv2.flood_fill() wrapper

    This function uses cv2 and numpy, is slow and not differentiable
    See https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill#floodfill
    See https://stackoverflow.com/questions/19839947/flood-fill-in-python

    Args:
        img (torch.Tensor): pytorch Tensor, dtype should be int. Support (H,W)
        target_value (int): Value that need to change
        fill_value (int): target_value will change to this value
        start_point (tuple(int, int)): starting point for flood fill.
        flags (int, optional):
          4 means that the four nearest neighbor pixels. (vertically and horizontally)
          8 means that the eight nearest neighbor pixels. (vertically, horizontally and diagonally)

    Returns:
        torch.Tensor: same shape as input.
    """
    assert len(start_point) == 2, 'Starting point should be a tuple of length 2 (x, y)'
    matrix_np = np.asarray(img).copy()
    numeric_matrix = np.where(matrix_np == target_value, 255, 0).astype(np.uint8)
    mask = np.zeros(np.asarray(numeric_matrix.shape) + 2, dtype=np.uint8)
    cv2.floodFill(numeric_matrix, mask, start_point, 255, flags=flags)
    mask = mask[1:-1, 1:-1]
    matrix_np[mask == 1] = fill_value
    return torch.tensor(matrix_np, dtype=img.dtype, device=img.device)
