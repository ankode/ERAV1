import albumentations as A
import numpy as np
from torchvision import datasets
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


a_train_transforms = A.Compose([
    A.PadIfNeeded(min_height=36, min_width=36, always_apply=True, p=1),
    A.RandomCrop(height=32, width=32, always_apply=True, p=4),
    # A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16,always_apply=False,fill_value=(0.5, 0.5, 0.5)),
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
    train = datasets.CIFAR10('./data', train=True, download=True, transform=get_augmentation(a_train_transforms))
    test = datasets.CIFAR10('./data', train=False, download=True, transform=get_augmentation(a_test_transforms))
    return train, test

def get_lr(optimizer):
    """
        For tracking how the learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def get_incorrect_images(model,test_loader, device, n=10):
    incorrect_images = []
    predicted_labels = []
    correct_labels = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        incorrect_items = pred.ne(target.view_as(pred))
        incorrect_indices = incorrect_items.view(-1).nonzero().view(-1)
        predicted_labels.extend([item.item() for item in pred[incorrect_indices[:n-len(incorrect_images)]]])
        correct_labels.extend([item.item() for item in target.view_as(pred)[incorrect_indices[:n-len(incorrect_images)]]])
        incorrect_images.extend([item for item in data[incorrect_indices[:n-len(incorrect_images)]]])
        if len(incorrect_images)==n:
            break
    return incorrect_images,predicted_labels,correct_labels

def imshow(img):
    img = img / 2 + 0.5     # Unnormalize
    npimg = img
    npimg = np.clip(npimg, 0, 1)  # Add this line to clip the values
    return np.transpose(npimg, (1, 2, 0))  # Convert from Tensor image


def plot_incorrect_images(model, test_loader, device, n):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))

    incorrect_images,predicted_labels,correct_labels = get_incorrect_images(model,test_loader, device, n)

    for i, image_tensor in enumerate(incorrect_images):
        ax = axes[i // 5, i % 5]  # Get the location of the subplot
        image = image_tensor.cpu().numpy()
        ax.imshow(imshow(image))  # Display the image
        ax.set_title(f"Predicted {class_names[predicted_labels[i]]}, Actual {class_names[correct_labels[i]]}")  # Set the title as the index

    plt.tight_layout()  # To provide sufficient spacing between subplots
    plt.show()

    return incorrect_images, predicted_labels, correct_labels

import matplotlib.pyplot as plt
def plot_train_test_loss(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot([x.cpu().item() for x in train_losses])
    axs[0, 0].set_title("Training Loss")
    axs[1,0].axis(ymin=0,ymax=100)
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1,1].axis(ymin=0,ymax=100)
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")


def plot_gcam_incorrect_preds(model,correct_labels, incorrect_images, predicted_labels):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))

    for i, image_tensor in enumerate(incorrect_images):
        ax = axes[i // 5, i % 5]  # Get the location of the subplot
        image = image_tensor.cpu().numpy()
        grayscale_cam = cam(input_tensor=image_tensor.reshape(1,3,32,32), targets=[ClassifierOutputTarget(predicted_labels[i])],aug_smooth=True,eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(imshow(image), grayscale_cam, use_rgb=True,image_weight=0.6)
        #ax.imshow(np.transpose(imshow(visualization), (2, 0, 1)))  # Display the image
        ax.imshow(visualization,interpolation='bilinear')
        ax.set_title(f"Predicted {class_names[predicted_labels[i]]}, Actual {class_names[correct_labels[i]]}")  # Set the title as the index

    plt.tight_layout()  # To provide sufficient spacing between subplots
    plt.show()

