import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class RotatedMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize([28, 28])]
        )

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.transforms(self.images[index])
        return image, label

    def __len__(self):
        return len(self.labels)


def mnist():
    data_path = "../../data/external/corruptmnist"

    train_imgs = None
    train_labels = None

    test_imgs = None
    test_labels = None

    for file in os.listdir(data_path):
        npz_obj = np.load(os.path.join(data_path, file))
        if "train" in file:
            if train_imgs is None:
                train_imgs = npz_obj["images"]
                train_labels = npz_obj["labels"]
            train_imgs = np.concatenate((train_imgs, npz_obj["images"]))
            train_labels = np.concatenate((train_labels, npz_obj["labels"]))
        elif "test" in file:
            test_imgs = npz_obj["images"]
            test_labels = npz_obj["labels"]

    trainset = RotatedMNISTDataset(train_imgs, train_labels)
    testset = RotatedMNISTDataset(test_imgs, test_labels)

    return trainset, testset
