import os
import sys
import torch

from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Callable, Optional

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import xml.etree.ElementTree as ET # changed
from PIL import ImageFile
from torchvision.transforms.functional import resize
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


def image_transform(
    image_size: Union[int, List[int]],
    augmentation: dict,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]) -> Callable:
    """Image transforms.
    """

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)

    # data augmentations
    horizontal_flip = augmentation.pop('horizontal_flip', None)
    if horizontal_flip is not None:
        assert isinstance(horizontal_flip, float) and 0 <= horizontal_flip <= 1

    vertical_flip = augmentation.pop('vertical_flip', None)
    if vertical_flip is not None:
        assert isinstance(vertical_flip, float) and 0 <= vertical_flip <= 1

    random_crop = augmentation.pop('random_crop', None)
    if random_crop is not None:
        assert isinstance(random_crop, dict)

    center_crop = augmentation.pop('center_crop', None)
    if center_crop is not None:
        assert isinstance(center_crop, (int, list))

    if len(augmentation) > 0:
        raise NotImplementedError('Invalid augmentation options: %s.' % ', '.join(augmentation.keys()))

    t = [
        transforms.Resize(image_size) if random_crop is None else transforms.RandomResizedCrop(image_size[0], **random_crop),
        transforms.CenterCrop(center_crop) if center_crop is not None else None,
        transforms.RandomHorizontalFlip(horizontal_flip) if horizontal_flip is not None else None,
        transforms.RandomVerticalFlip(vertical_flip) if vertical_flip is not None else None,
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]

    return transforms.Compose([v for v in t if v is not None])


def fetch_data(
    dataset: Callable[[str], Dataset],
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    train_splits: List[str] = [],
    test_splits: List[str] = [],
    train_shuffle: bool = True,
    test_shuffle: bool = False,
    train_augmentation: dict = {},
    test_augmentation: dict = {},
    batch_size: int = 1,
    dataloader_flag: str = 'counting',
    test_batch_size: int = 1) -> Tuple[List[Tuple[str, DataLoader]], List[Tuple[str, DataLoader]]]:
    """
    currently, only support test_batch_size=1
    """

    # fetch training data
    train_transform = transform(augmentation=train_augmentation) if transform else None
    train_loader_list = []
    for split in train_splits:
        train_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split,
                transform = train_transform,
                target_transform = target_transform,
                dataloader_flag = dataloader_flag),
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = train_shuffle)))

    # fetch testing data
    test_transform = transform(augmentation=test_augmentation) if transform else None
    test_loader_list = []
    for split in test_splits:
        test_loader_list.append((split, DataLoader(
            dataset = dataset(
                split = split,
                transform = test_transform,
                target_transform = target_transform,
                dataloader_flag = dataloader_flag),
            batch_size = batch_size if test_batch_size is None else test_batch_size,
            num_workers = num_workers,
            pin_memory = pin_memory,
            drop_last=drop_last,
            shuffle = test_shuffle)))

    return train_loader_list, test_loader_list

def get_mean_std(dataset):
    """calc the mean std for an dataset

    Args:
        dataset (torch.utils.data.Dataset): pytorch Dataset. Remeber only put `ToTensor()` on transform

    Returns:
        (Tensor, Tensor): mean, std. Each with 3 values (RGB)

    Notes:
        PASCAL: 
          448 - [0.4472, 0.4231, 0.3912], [0.2358, 0.2295, 0.2324]
          368 - [0.4472, 0.4231, 0.3912], [0.2350, 0.2287, 0.2316]
          224 - [0.4472, 0.4231, 0.3912], [0.2312, 0.2249, 0.2279]
        COCO:
          448 - [0.4714, 0.4475, 0.4078], [0.2382, 0.2332, 0.2363]
          368 - [0.4713, 0.4474, 0.4077], [0.2370, 0.2319, 0.2351]
          224 - [0.4713, 0.4474, 0.4077], [0.2330, 0.2279, 0.2313]
    """
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=2)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _, _ in train_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def F_multi_resolution(resize, transform=None):
    if not isinstance(resize, list):
        raise ValueError("F_multi_resolution is ON, resize should be a list")
    if isinstance(transform, transforms.Compose):
        raise ValueError("multi_resolution is ON, transform should be List[transform.]")
    if isinstance(transform, list) and any(isinstance(x, transforms.Compose) for x in transform):
        raise ValueError("multi_resolution is ON, transform should be List[transform.]")
    if isinstance(transform, list) and any(isinstance(x, list) for x in transform):
        raise ValueError("multi_resolution is ON, transform should be List[transform.]")
    if isinstance(transform, list) and not any(isinstance(x, transforms.ToTensor) for x in transform):
        raise ValueError("multi_resolution is ON, transform should contains one transforms.ToTensor")
    if not transform:
        transform_list = []
        for size in resize:
            if isinstance(size, int):
                size = (size, size)
            if not isinstance(size, tuple):
                raise ValueError("multi_transform is ON, resize should be List[int] or List[tuple]")
            transform_list.append(
                transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))
                ]))
        transform = transforms.Compose([]), transform_list
    else:
        if not isinstance(transform[0], transforms.Compose):
            if any(isinstance(x, transforms.Compose) for x in transform):
                raise ValueError("transform should NOT contains transforms.Compose. Expected transform to be List[transforms.]")
            transform_list = []
            split=0
            for i in range(len(transform)):
                if isinstance(transform[i], transforms.ToTensor):
                    split=i
            aug_transform = transforms.Compose(transform[:split])
            tensor_transform = transform[split:]
            for size in resize:
                if isinstance(size, int):
                    size = (size, size)
                if not isinstance(size, tuple):
                    raise ValueError("multi_transform is ON, resize should be List[int] or List[tuple]")
                transform_list.append(
                    transforms.Compose([transforms.Resize(size)] +tensor_transform)
                    )
            transform = aug_transform, transform_list
        elif isinstance(transform[0], transforms.Compose):
            if any(not isinstance(x, transforms.Compose) for x in transform):
                raise ValueError("transform should NOT contains transforms.Compose. Expected transform to be List[transforms.]")
    return transform

def F_mutli_transform(resize, transform=None):
    if not isinstance(resize, list) and isinstance(transform, list) and not isinstance(transform[0], transforms.Compose):
        raise ValueError("multi_transform is ON, resize should be a list OR transform should be List[transforms.Compose]")
    if isinstance(transform, transforms.Compose):
        raise ValueError("multi_transform is ON, transform should be either List[transform.] or List[transform.Compose]")
    if isinstance(transform, list) and any(isinstance(x, list) for x in transform):
        raise ValueError("multi_transform is ON, transform should be either List[transform.] or List[transform.Compose]")
    if not transform:
        transform_list = []
        for size in resize:
            if isinstance(size, int):
                size = (size, size)
            if not isinstance(size, tuple):
                raise ValueError("multi_transform is ON, resize should be List[int] or List[tuple]")
            transform_list.append(
                transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))
                ]))
        transform = transform_list
    elif isinstance(transform, list):
        if not isinstance(transform[0], transforms.Compose):
            if any(isinstance(x, transforms.Compose) for x in transform):
                raise ValueError("transform contains both `transforms.Compose` and `transforms.` Expected to contain only one of them")
            transform_list = []
            for size in resize:
                if isinstance(size, int):
                    size = (size, size)
                if not isinstance(size, tuple):
                    raise ValueError("multi_transform is ON, resize should be List[int] or List[tuple]")
                transform_list.append(
                    transforms.Compose([transforms.Resize(size)] +transform)
                    )
            transform = transform_list
        elif isinstance(transform[0], transforms.Compose):
            if any(not isinstance(x, transforms.Compose) for x in transform):
                raise ValueError("transform contains both `transforms.Compose` and `transforms.` Expected to contain only one of them")
            pass
    return transform


def pascal_voc_object_categories(query: Optional[Union[int, str]] = None) -> Union[int, str, List[str]]:
    """PASCAL VOC dataset class names.
    """

    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

    if query is None:
        return categories
    else:
        for idx, val in enumerate(categories):
            if isinstance(query, int) and idx == query:
                return val
            elif val == query:
                return idx


class VOC_Classification(Dataset):
    """Dataset for PASCAL VOC.
    """

    def __init__(self, data_dir, dataset, split, classes, dataloader_flag, transform=None, target_transform=None, multi_resolution=False,  multi_transform=False):
        self.data_dir = data_dir
        self.dataset = dataset
        self.split = split
        self.image_dir = os.path.join(data_dir, dataset, 'JPEGImages')
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.gt_path = os.path.join(self.data_dir, self.dataset, 'ImageSets', 'Main')
        assert os.path.isdir(self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.multi_resolution = multi_resolution
        self.multi_transform = multi_transform
        if dataloader_flag=='counting':
            ## counting training dataloader
            self.image_labels = self._read_annotations_07_regression(self.split)
        elif dataloader_flag=='ins_seg':
            ## instance segmentation training dataloader
            self.image_labels = self._read_annotations_124_plus_segtrain_rm_val_regression(self.split)
        else:
            print("error, dataloader_flag is neither counting or ins_seg")

        print("number of images for %s: %d" %(split,len(self.image_labels)))

    def _read_annotations_124_plus_segtrain_rm_val_regression(self, split):
        class_labels = OrderedDict()
        num_classes = len(self.classes)
        dim=14
        with open(self.poc_path+'/Data/Datasets/Pascal_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt','r') as f:
            val_list=[]
            for ima in f:
                ima=ima.strip('\n')
                val_list.append(ima)
        with open('/media/guolei/DISK1TB/Datasets/SBD/trainval.txt','r') as f:
            val_list2=[]
            for ima in f:
                ima=ima.strip('\n')
                val_list2.append(ima)
        if os.path.exists(os.path.join(self.gt_path, split + '.txt')):
            for class_idx in range(num_classes):
                filename = os.path.join(
                    self.gt_path, self.classes[class_idx] + '_' + split + '.txt')
                with open(filename, 'r') as f:
                    for line in f:
                        name, label = line.split()
                        if (name not in val_list) and ('2007_'+name not in val_list) and (name in val_list2):
                            if name not in class_labels:
                                class_labels[name] = [np.zeros(num_classes),np.zeros((num_classes,dim,dim),dtype=int)]
                            if int(label)!=-1:
                                count=self.return_count_obj(os.path.join(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/Annotations/',name+'.xml'),
                                    self.classes[class_idx])
                                mask_obj=self.return_mask_obj(os.path.join(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/Annotations/',name+'.xml'),
                                    self.classes[class_idx],dim)
                                if count!=np.sum(mask_obj):
                                    print(count,np.sum(mask_obj))
                                    import matplotlib.pyplot as plt
                                    plt.imshow(mask_obj)
                                    print(name)
                                    print(mask_obj)
                                    print('error')

                                class_labels[name][0][class_idx] = int(count)
                                class_labels[name][1][class_idx] = mask_obj
                            else:
                                class_labels[name][0][class_idx] = int(0)
        if os.path.exists(os.path.join(self.poc_path+'/Datasets/Pascal_2007/VOCdevkit/VOC2007/ImageSets/Main/', split + '.txt')):
            #for class_idx in range(num_classes):
            #   filename = os.path.join(
            #      '/media/rao/Data/Datasets/Pascal_2007/VOCdevkit/VOC2007/ImageSets/Main/', self.classes[class_idx] + '_' + split + '.txt')
            with open(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', 'r') as f:
                for line in f:
                    name = line.strip('\n')
                    if (name not in val_list) and ('2007_'+name not in val_list):
                        if name not in class_labels:
                            class_labels[name] = [np.zeros(num_classes),np.zeros((num_classes,dim,dim),dtype=int)]
                            # class_labels[name][class_idx] = int(label)
                            class_labels[name][0]=self.return_segtrain_gt(num_classes,name)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.dataset))

        return list(class_labels.items())

    def _read_annotations_07_regression(self, split):
        class_labels = OrderedDict()
        num_classes = len(self.classes)
        if os.path.exists(os.path.join(self.data_dir,self.dataset,'ImageSets/Main/', split + '.txt')):
            for class_idx in range(num_classes):
                filename = os.path.join(
                    self.data_dir,self.dataset,'ImageSets/Main/', self.classes[class_idx] + '_' + split + '.txt')
                with open(filename, 'r') as f:
                    for line in f:
                        name, label = line.split()
                        if name not in class_labels:
                            class_labels[name] = np.zeros(num_classes)
                        class_labels[name][class_idx] = int(label)
                        if int(label)!=-1:
                            count=self.return_count_obj_rm_diff(os.path.join(self.data_dir,self.dataset,'Annotations',name+'.xml'),
                                self.classes[class_idx])
                            class_labels[name][class_idx] = int(count)
                        else:
                            class_labels[name][class_idx] = int(0)
        else:
            raise NotImplementedError(
                'Invalid "%s" split for PASCAL %s classification task.' % (split, self.dataset))

        return list(class_labels.items())

    def return_count_obj_rm_diff(self,xml_file,class_name):
        count=0
        tree = ET.parse(xml_file)
        objs = tree.findall('object')
        for ix, obj in enumerate(objs):
            if obj.find('name').text==class_name and int(obj.find('difficult').text)==0:
                count+=1
        return count

    def return_segtrain_gt(self,num_classes,ima):
        gt_one=np.zeros(num_classes,dtype=int)
        im = Image.open(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/SegmentationObject/'+ima+'.png') # Replace with your image name here
        indexed = np.int64(np.array(im))
        im2 = Image.open(self.poc_path+'/Datasets/Pascal_2012/VOCdevkit/VOC2012/SegmentationClass/'+ima+'.png') # Replace with your image name here
        indexed2 = np.int64(np.array(im2))
        indexed[indexed==255]=-1
        indexed2[indexed2==255]=-1
        uniq_ins=set(list(indexed.flatten()))
        for i in uniq_ins:
            if i >0:
                ins=indexed==i
                x,y=np.where(ins==True)
                label=indexed2[x[0],y[0]]
                gt_one[int(label)-1]+=1
        return gt_one

    def return_mask_obj(self,xml_file,class_name,dim):
        # dim=14
        mask_obj=np.zeros((dim,dim), dtype =int)
        tree = ET.parse(xml_file)
        objs = tree.findall('object')
        size = tree.findall('size')
        # print(size)
        width=float(size[0].find('width').text)
        height=float(size[0].find('height').text)
        for ix, obj in enumerate(objs):
            if obj.find('name').text.lower().strip()==class_name:
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)-1
                y1 = float(bbox.find('ymin').text)-1
                x2 = float(bbox.find('xmax').text)-1
                y2 = float(bbox.find('ymax').text)-1
                mask_obj[int(np.round((x1+x2)/2*dim/width-0.5)),int(np.round((y1+y2)/2*dim/height-0.5))]+=1
        return mask_obj

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        if len(target)==2:
            target0=target[0]
            target1=target[1]
        elif isinstance(target,np.ndarray):
            target0=target
            target1=np.array([1])

        target0 = torch.from_numpy(target0).float()
        # print(type(1*target1))
        target1 = torch.from_numpy(1*target1).float()
        # target = torch.from_numpy(target).float()
        img = Image.open(os.path.join(
            self.image_dir, filename + '.jpg')).convert('RGB')
        if self.multi_transform:
            imgs = []
            if self.transform:
                for trans in self.transform:
                    imgs.append(trans(img))
            if self.target_transform:
                target0 = self.target_transform(target0)
            return tuple(imgs), target0, target1
        elif self.multi_resolution:
            imgs = []
            org_size = img.size[1] / img.size[0]#PLT.size is (x-axis, y_axis). np.shape is (y-axis, x_axis)
            resized_size = []
            if len(self.transform) == 2:
                img = self.transform[0](img)
                for trans in self.transform[1]:
                    imgs.append(trans(img))
                    resized_size.append(imgs[-1].shape[-2]/imgs[-1].shape[-1])
                one_hot_idx = np.argmin(abs(np.asarray(resized_size) - org_size))
                one_hot = torch.zeros(len(resized_size))
                one_hot[one_hot_idx] = 1.
                target1 = one_hot
            else:
                raise ValueError(
                    "transform should be (transforms.Compose, List[transforms.Compose])"
                )
            if self.target_transform:
                target0 = self.target_transform(target0)
            return tuple(imgs), target0, target1
        else:
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                target0 = self.target_transform(target0)
            return img, target0,target1

    def __len__(self):
        return len(self.image_labels)

def pascal_voc_classification(
        split: str,
        data_dir: str,
        year: int = 2007,
        dataloader_flag: str = 'counting',
        resize: int = 448,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        multi_resolution: bool = False,
        multi_transform: bool = False) -> Dataset:
    """PASCAL VOC dataset.
    data_dir: usually `dataset/pascal_2007/VOCdevkit`

    Args:
        multi_resolution (bool): Perform exactly the SAME (random) transform but with different resize 
          if True -> resize should be a List[] of resolution (diff size)
                     transform should be List[transform.] that contains one transforms.Tensor. 
                     All random transformation should be put before transforms.Tensor.
                     Each resolution (in resize) will be broadcase to ``List[transform.]``
          Normal Usage: turn this on and set resize to a list.
          If this is True -> internal transform will become (transforms.Compose, List[transforms.Compose])
        mutli_transform (bool): Given a list of transform.Compose, apply them on each img. Notice: each img will have different random_transform.
          if True -> resize should be a List[] of resolution (diff size)
                     or transform should be a List[transforms.Compose] or List[transform.]
                     If given ``List[transform.]``. Each resolution (in resize) will be broadcase to ``List[transform.]``
                     If given ``List[transforms.Compose]``, Override the ``resize`` parameter.
          Normal Usage: turn this on and transfrom = List[transforms.Compose].
          If this is True -> internal transform will become List[transforms.Compose]
        
    Returns:
        Dataset: torch.datasets, Noramlly.
          It contains three thing for each data point(batch=1)
            - (BATCH, C, H, W), image
            - (BATCH, Class), number of objection in class_i
            - (BATCH, scalar=1), No idea what is this
        Dataset: torch.datasets, if multi_resolution is ONE
          It contains three thing for each data point(batch=1)
            - (BATCH, C, H, W), image
            - (BATCH, Class), number of objection in class_i
            - (BATCH, len(resize)), One-hot vector that represent which resized image is more close to the original image.
              For example: org imge size is (400, 600). Resized images are [(448,488),(321, 448),(448,327)]. This returns [0,1,0]
                           org imge size is (380, 420). Resized images are [(448,488),(321, 448),(448,327)]. This returns [1,0,0]
        
    """
    if not multi_transform and not multi_resolution and not isinstance(resize, int) and not isinstance(resize, tuple):
        raise ValueError("resize should be a int or tuple")
    if not multi_transform and not multi_resolution and isinstance(transform, list) and isinstance(transform[0], transforms.Compose):
        raise ValueError("transform should be either transforms.Compose or List[transforms.]. Not List[transforms.Compose]")

    if isinstance(resize, int):
        resize = (resize, resize)
    if multi_transform:
        transform = F_mutli_transform(resize, transform)
    elif multi_resolution:
        transform = F_multi_resolution(resize, transform)
    else:
        if not transform:
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif isinstance(transform, list):
            transform = transforms.Compose([transforms.Resize(resize)] + transform)
    object_categories = pascal_voc_object_categories()
    dataset = 'VOC' + str(year)
    # print(transform)
    return VOC_Classification(data_dir, dataset, split, object_categories,dataloader_flag, transform, target_transform, multi_resolution=multi_resolution, multi_transform=multi_transform)

def get_pascal_class_dict():
    return dict(
        zip(range(20), [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]))

def get_COCO_class_dict():
    """Get dict of class name in order
        """
    return {0: u'__background__',
    1: u'person',
    2: u'bicycle',
    3: u'car',
    4: u'motorcycle',
    5: u'airplane',
    6: u'bus',
    7: u'train',
    8: u'truck',
    9: u'boat',
    10: u'traffic light',
    11: u'fire hydrant',
    12: u'stop sign',
    13: u'parking meter',
    14: u'bench',
    15: u'bird',
    16: u'cat',
    17: u'dog',
    18: u'horse',
    19: u'sheep',
    20: u'cow',
    21: u'elephant',
    22: u'bear',
    23: u'zebra',
    24: u'giraffe',
    25: u'backpack',
    26: u'umbrella',
    27: u'handbag',
    28: u'tie',
    29: u'suitcase',
    30: u'frisbee',
    31: u'skis',
    32: u'snowboard',
    33: u'sports ball',
    34: u'kite',
    35: u'baseball bat',
    36: u'baseball glove',
    37: u'skateboard',
    38: u'surfboard',
    39: u'tennis racket',
    40: u'bottle',
    41: u'wine glass',
    42: u'cup',
    43: u'fork',
    44: u'knife',
    45: u'spoon',
    46: u'bowl',
    47: u'banana',
    48: u'apple',
    49: u'sandwich',
    50: u'orange',
    51: u'broccoli',
    52: u'carrot',
    53: u'hot dog',
    54: u'pizza',
    55: u'donut',
    56: u'cake',
    57: u'chair',
    58: u'couch',
    59: u'potted plant',
    60: u'bed',
    61: u'dining table',
    62: u'toilet',
    63: u'tv',
    64: u'laptop',
    65: u'mouse',
    66: u'remote',
    67: u'keyboard',
    68: u'cell phone',
    69: u'microwave',
    70: u'oven',
    71: u'toaster',
    72: u'sink',
    73: u'refrigerator',
    74: u'book',
    75: u'clock',
    76: u'vase',
    77: u'scissors',
    78: u'teddy bear',
    79: u'hair drier',
    80: u'toothbrush'}

def class2label(arr, dataset='coco'):
    """Turn class_idx to class label with counting.
    Input should be something like [[3.,-0., 0.],[-0.,3.,2.]]
    Output is something like [{"car":3}, {"person":3, "plane":2}]

    Args:
        arr (np.ndarray): either 1D(C) or 2D(batch, C) array.
          Values should the counting for that obj_idx
        dataset (str): either 'coco' or 'pascal'

    Returns:
        dict or list(dict): 
          - dict if Input is 1D(C): dict(class_label: countint)
          - list(dict) if Input is 2D(batch, C).
    """
    if dataset.lower() == 'pascal':
        categories = np.asarray([
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'])
    elif dataset.lower() == 'coco':
        categories = np.asarray(list(get_COCO_class_dict().values())[1:])
    else:
        raise ValueError(f"Not support dataset `{dataset}`")

    def testing(idx, classes):
        if not isinstance(classes, np.ndarray):
            classes = np.asarray(classes)
        if len(idx.shape) == 1:
            loc = np.where(idx > 0)
            return dict(zip(list(categories[loc]), list(idx[loc])))
        elif len(idx.shape) == 2:
            out = []
            for inner in idx:
                out.append(testing(inner, classes))
            return out
        else:
            raise ValueError("Not support arr.dim is larger than 2")
    if isinstance(arr, torch.Tensor):
        return testing(arr.cpu(), categories)
    elif isinstance(arr, np.ndarray):
        return testing(arr, categories)
    elif hasattr(arr, '__iter__'):
        return testing(np.asarray(arr), categories)

class coco_Classification(Dataset):
    """ Helps to turn COCO dataset into torch.utils.data.Dataset class
    """
    def __init__(self, data_dir,split,year,dataloader_flag, transform=None, target_transform=None, multi_resolution=False, multi_transform=False):
        self.data_dir = data_dir
        self.split = split
        self.year = year
        ## data_dir: /media/rao/Data/Datasets/MSCOCO/coco/
        self.image_dir = os.path.join(data_dir,split+str(year))
        assert os.path.isdir(self.image_dir), 'Could not find image folder "%s".' % self.image_dir
        self.gt_path = os.path.join(self.data_dir, 'annotations')
        assert os.path.isdir(self.gt_path), 'Could not find ground truth folder "%s".' % self.gt_path
        self.transform = transform
        self.target_transform = target_transform
        self.cocoInstance = None
        self.multi_resolution = multi_resolution
        self.multi_transform = multi_transform
        if dataloader_flag=='counting':
            ## use coco 2017 train data
            self.image_labels = self._read_annotations(split,year)
        else:
            print('error, dataloader_flag should be counting')
        if split=='val':
            index=int(len(self.image_labels)/2)
            self.image_labels=self.image_labels[:index]
        print(f"There are {len(self.image_labels)} datapoint")

    def _read_annotations(self,split,year):
        gt_file=os.path.join(self.gt_path,'instances_'+split+str(year)+'.json')
        cocoGt=COCO(gt_file)
        # Usage: https://blog.csdn.net/qq_41185868/article/details/103897159
        self.cocoInstance = cocoGt
        catids=cocoGt.getCatIds()
        num_classes=len(catids)
        catid2index={}
        for i,cid in enumerate(catids):
            catid2index[cid]=i
        annids=cocoGt.getAnnIds()
        class_labels = OrderedDict()
        for id in annids:
            anns=cocoGt.loadAnns(id)
            for i in range(len(anns)):
                ann=anns[i]
                name=ann['image_id']
                if name not in class_labels:
                    class_labels[name]=np.zeros(num_classes)
                category_id=ann['category_id']
                class_labels[name][catid2index[category_id]]+=1
        return list(class_labels.items())

    def loadCats(self, id):
        """Map id to class name

        Returns:
            dict: {supercategory, id, name}
        """
        if self.cocoInstance is not None:
            return self.cocoInstance.loadCats(id)

    def __getitem__(self, index):
        filename, target = self.image_labels[index]
        target0=target
        target1=np.array([1])
        target0 = torch.from_numpy(target0).float()
        # print(type(1*target1))
        target1 = torch.from_numpy(1*target1).float()
        # target = torch.from_numpy(target).float()
        # 000000291625.jpg
        filename='0'*(12-len(str(filename)))+str(filename)
        # img = Image.open(os.path.join(
        #     self.image_dir, 'COCO_train2014_'+ filename + '.jpg')).convert('RGB')
        img = Image.open(os.path.join(
            self.image_dir, f"COCO_{self.split}{self.year}_{filename}.jpg")).convert('RGB')
        if self.multi_transform:
            imgs = []
            if self.transform:
                for trans in self.transform:
                    imgs.append(trans(img))
            if self.target_transform:
                target0 = self.target_transform(target0)
            return tuple(imgs), target0, target1
        elif self.multi_resolution:
            imgs = []
            org_size = img.size[1] / img.size[0]#PLT.size is (x-axis, y_axis). np.shape is (y-axis, x_axis)
            resized_size = []
            if len(self.transform) == 2:
                img = self.transform[0](img)
                for trans in self.transform[1]:
                    imgs.append(trans(img))
                    resized_size.append(imgs[-1].shape[-2]/imgs[-1].shape[-1])
                one_hot_idx = np.argmin(abs(np.asarray(resized_size) - org_size))
                one_hot = torch.zeros(len(resized_size))
                one_hot[one_hot_idx] = 1.
                target1 = one_hot
            else:
                raise ValueError(
                    "transform should be (transforms.Compose, List[transforms.Compose])"
                )
            if self.target_transform:
                target0 = self.target_transform(target0)
            return tuple(imgs), target0, target1
        else:
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                target0 = self.target_transform(target0)
            return img, target0, target1

    def __len__(self):
        return len(self.image_labels)


def coco_classification(
    split: str,
    data_dir: str,
    year: int = 2014,
    dataloader_flag: str = 'counting',
    resize: int = 448,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    multi_resolution: bool = False,
    multi_transform: bool = False) -> Dataset:
    """Create a torch.utils.data.Dataset Class from COCO dataset

    Args:
        split (str): either "train" or "test"
        data_dir (str): coco dataset path. usually `dataset/COCO`
        year (int, optional): coco dataset year. Defaults to 2014.
        dataloader_flag (str, optional): Useless. Defaults to 'counting'.
        resize (int or tuple, optional): Resize the input image.
          if int -> resize to square, equal side
        transform (Optional[Callable], optional): transformations of dataset. Defaults to None.
          Accept List or transforms.Compose(). If using transforms.Compose(), will override resize.
          Meaning if you using transform.Compose(), rmb the put resize mannually and be consistent with val_set
        target_transform (Optional[Callable], optional): transformations of target. Defaults to None.
        multi_resolution (bool): Perform exactly the SAME (random) transform but with different resize 
          if True -> resize should be a List[] of resolution (diff size)
                     transform should be List[transform.] that contains one transforms.Tensor. 
                     All random transformation should be put before transforms.Tensor.
                     Each resolution (in resize) will be broadcase to ``List[transform.]``
          Normal Usage: turn this on and set resize to a list.
          If this is True -> internal transform will become (transforms.Compose, List[transforms.Compose])
        mutli_transform (bool): Given a list of transform.Compose, apply them on each img. Notice: each img will have different random_transform.
          if True -> resize should be a List[] of resolution (diff size)
                     or transform should be a List[transforms.Compose] or List[transform.]
                     If given ``List[transform.]``. Each resolution (in resize) will be broadcase to ``List[transform.]``
                     If given ``List[transforms.Compose]``, Override the ``resize`` parameter.
          Normal Usage: turn this on and transfrom = List[transforms.Compose].
          If this is True -> internal transform will become List[transforms.Compose]

    Returns:
        Dataset: torch.datasets.
          It contains three thing for each data point(batch=1)
            - (BATCH, C, H, W), image
            - (BATCH, Class), number of objection in class_i
            - (BATCH, scalar=1), No idea what is this
    """
    if not multi_transform and not multi_resolution and not isinstance(resize, int) and not isinstance(resize, tuple):
        raise ValueError("resize should be a int or tuple")
    if not multi_transform and not multi_resolution and isinstance(transform, list) and isinstance(transform[0], transforms.Compose):
        raise ValueError("transform should be either transforms.Compose or List[transforms.]. Not List[transforms.Compose]")

    if isinstance(resize, int):
        resize = (resize, resize)
    if multi_transform:
        transform = F_mutli_transform(resize, transform)
    elif multi_resolution:
        transform = F_multi_resolution(resize, transform)
    else:
        if not transform:
            transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif isinstance(transform, list):
            transform = transforms.Compose([transforms.Resize(resize)] + transform)
    # print(transform)
    return coco_Classification(data_dir, split,year,dataloader_flag, transform, target_transform,  multi_resolution=multi_resolution, multi_transform=multi_transform)