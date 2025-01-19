import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class CellData(Dataset):
    def __init__(self, data_dir, size, train='True', train_test_split=0.8, augment_data=True):
        ##########################inputs##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(CellData, self).__init__()
        # todo
        # initialize the data class
        self.data_dir = data_dir
        self.size = size
        self.train = train
        self.train_test_split = train_test_split
        self.augment_data = augment_data

        self.images = os.listdir(os.path.join(self.data_dir, 'scans'))
        self.masks = os.listdir(os.path.join(self.data_dir, 'labels'))

    def __getitem__(self, idx):
        # todo
        # load image and mask from index idx of your data
        img_path = os.getcwd() + "/data/cells/scans/" + self.images[idx]
        mask_path = os.getcwd() + "/data/cells/labels/" + self.masks[idx]

        # convert image to grayscale
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # image resize - ToTensor() will normalize img to [0, 1] and convert to Tensor
        transform_img = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor()
        ])
        image = transform_img(image)
        # mask resize - mask is already [0, 1], PILToTensor() will only convert it to Tensor without changing its values
        transform_mask = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.PILToTensor()
        ])
        mask = transform_mask(mask)

        # data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 4)
            if augment_mode == 0:
                # todo
                # flip image vertically
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])
            elif augment_mode == 1:
                # todo
                # flip image horizontally
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            elif augment_mode == 2:
                # todo
                # rotate image
                angle = np.random.randint(0, 360)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            else:
                # todo
                # apply gamma correction to image
                image = TF.adjust_gamma(image, gamma=1.5)
                mask = TF.adjust_gamma(mask, gamma=1.5)

        # todo
        # return image and mask in tensors
        return image, mask

    def __len__(self):
        return len(self.images)
