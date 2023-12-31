import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, Compose, InterpolationMode
from PIL import Image
import os
import matplotlib.pyplot as plt

class UNetDataClass(Dataset):
    def __init__(self, images_path, masks_path, transform):
        super(UNetDataClass, self).__init__()
        
        images_list = os.listdir(images_path)
        masks_list = os.listdir(masks_path)
        
        images_list = [images_path + image_name for image_name in images_list]
        masks_list = [masks_path + mask_name for mask_name in masks_list]
        
        self.images_list = images_list
        self.masks_list = masks_list
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]
        
        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)
        
        # Normalize
        data = self.transform(data)
        label = self.transform(label)
        
        label = torch.where(label>0.65, 1.0, 0.0)
        
        label[2, :, :] = 0.0001
        label = torch.argmax(label, 0).type(torch.int64)
        
        return data, label
    
    def __len__(self):
        return len(self.images_list)

if __name__ == "__main__":
    images_path = "./data/train/train/"
    masks_path =  "./data/train_gt/train_gt/"
    transform = Compose([Resize((800, 1120), interpolation=InterpolationMode.BILINEAR),
                     PILToTensor()])
    unet_dataset = UNetDataClass(images_path, masks_path, transform)
    print(f"[INFO] Size of dataset: {len(unet_dataset)}")
    train_size = 0.8
    valid_size = 0.2
    batch_size = 8
    train_set, valid_set = random_split(unet_dataset, 
                                    [int(train_size * len(unet_dataset)) , 
                                     int(valid_size * len(unet_dataset))])
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    
    print(f"[INFO] Train dataloader: {len(train_dataloader)} batches")
    print(f"[INFO] Valid dataloader: {len(valid_dataloader)} batches")

    for i, (data,targets) in enumerate(train_dataloader):
        img = data
        mask = targets
        break

    print(f"[INFO] Dimesions of train images: {img.size()}")
    print(f"[INFO] Dimesions of valid images: {mask.size()}")
    print(f"[INFO] Data type of train images: {img.type()}")
    print(f"[INFO] Data type of valid images: {mask.type()}") 

    plt.subplot(1, 2, 1)
    plt.imshow(img[0].permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(mask[0])
    plt.show()
