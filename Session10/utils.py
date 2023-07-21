import albumentations as A
import numpy as np
from torchvision import datasets
from albumentations.pytorch import ToTensorV2

a_train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,
                    always_apply=False,
                    # fill_value=tuple(np.mean(datasets.CIFAR10(root='./data', train=True, download=True).data, axis=(0, 1)))),
                    fill_value=(0.5, 0.5, 0.5)),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

a_test_transforms = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])


def get_augmentation(transforms):
    return lambda img: transforms(image=np.array(img))['image']


def get_mnist_data():
    train = datasets.CIFAR10('./data', train=True, download=True, transform=get_augmentation(train_transforms))
    test = datasets.CIFAR10('./data', train=False, download=True, transform=get_augmentation(test_transforms))

def get_lr(optimizer):
    """
        For tracking how the learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


