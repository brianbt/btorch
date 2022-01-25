import torch
import numpy as np
import warnings
import cv2

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
        A=A.transpose(1,2,0)
    if len(A.shape) == 3:
        #turn to grayscale
        A=np.dot(A[...,:3], [0.299, 0.587, 0.114])
    toFind = np.unique(A)
    out=dict()
    for tar in toFind:
        if tar == 0:
            continue
        i,j = np.where(A==tar)
        topLeft = (min(i), min(j))
        bottomRight = (max(i), max(j))
        if flexible:
            topLeft = (math.floor(topLeft[0]*(1-flexible)), math.floor(topLeft[1]*(1-flexible)))
            bottomRight = (math.floor(bottomRight[0]*(1+flexible)), math.floor(bottomRight[1]*(1+flexible)))
        out[tar]=[topLeft, bottomRight]
    return A, out

def replace_part(img, part, loc, handle_transparent=False):
    """replace part of `img` by `part`. It support batch mode.

    Args:
        img (Tensor or ndarray): either (N,C,H,W) or (C,H,W)
        part (Tensor or ndarray): used to replace. (C,H,W)
        loc (List[int]): [x_min, x_max], the topLeft coordinate.
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
            part_mask = part_mask.expand([img.shape[0]]+list(part.shape))
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
          it accepts,
            (List[List]): List of bbox if raw is False
            (List[Dict[]]): List of dictionary if raw is True. 
              This should be the default target output of datasets.COCODection()
        bbox_format (str, optional): Format of bboxs.
          Either `pascal` or `coco`. Defaults to 'pascal'.
        raw (bool, optional): see bboxs doc. Defaults to False.
        return_dict (bool, optional): only support when `raw` is ON

    Returns:
        List[Tensor]: List of Cropped Images
        of 
        Dict[category_id:[Tensor]]: Dict of List of Cropped Images for each class

    Example:
        >>> for img, target in datasets.COCODection():
        >>>     break
        >>> datasets.crop_img_from_bbx(i[0], i[1], bbox_format='coco', raw=True, return_dict=True)
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
                raise Exception("I should be a Dict")
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
        out =  dict(
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
        Dict[id:List[MASK]]: Each mask is a 2D matrix that contains either 0, 1. Size is same as `img`
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
