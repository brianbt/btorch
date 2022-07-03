import torch


def get_mean_std(dataset):
    """calc the mean std for an dataset

    Args:
        dataset (torch.utils.data.Dataset): pytorch Dataset. Remeber only put `ToTensor()` on transform

    Returns:
        (Tensor, Tensor): mean, std. Each with 3 values (RGB)

    Notes:
        IMAGENET:
          224 - [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        PASCAL: 
          448 - [0.4472, 0.4231, 0.3912], [0.2358, 0.2295, 0.2324]
          368 - [0.4472, 0.4231, 0.3912], [0.2350, 0.2287, 0.2316]
          224 - [0.4472, 0.4231, 0.3912], [0.2312, 0.2249, 0.2279]
        COCO:
          448 - [0.4714, 0.4475, 0.4078], [0.2382, 0.2332, 0.2363]
          368 - [0.4713, 0.4474, 0.4077], [0.2370, 0.2319, 0.2351]
          224 - [0.4713, 0.4474, 0.4077], [0.2330, 0.2279, 0.2313]
    """
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
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
