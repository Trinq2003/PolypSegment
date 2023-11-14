from torchsummary import summary
from torchgeometry.losses import one_hot
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import imageio
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
from collections import OrderedDict
import wandb

from data.dataloader import UNetDataClass
from model.encoder import EncoderBlock
from model.decoder import DecoderBlock
from model.bottleneck import BottleneckBlock
from model.unet import UNet
from model.CEDiceloss import CEDiceLoss

from utilities import utils, test, train, arg_parser

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = arg_parser.parser.parse_args()

# Set hyperparameters
num_classes = 3
epochs = args.num_epochs
learning_rate = args.lr
batch_size = args.batch_size
display_step = args.display_step
train_size = args.train_size
valid_size = args.valid_size

checkpoint_path = args.checkpoint_path
pretrained_path = args.pretrained_path
images_path = args.images_path
masks_path =  args.masks_path

loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []
train_loss_array = []
test_loss_array = []
last_loss = 9999999999999

# Dataloader
print(f"[PROGRESS] STEP 1: Loading data...")
transform = Compose([Resize((800, 1120), interpolation=InterpolationMode.BILINEAR),
                     PILToTensor()])
unet_dataset = UNetDataClass(images_path, masks_path, transform)
print(f"[INFO] Size of dataset: {len(unet_dataset)}")
train_set, valid_set = random_split(unet_dataset, 
                                [int(train_size * len(unet_dataset)) , 
                                    int(valid_size * len(unet_dataset))])
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

print(f"[INFO] Train dataloader: {len(train_dataloader)} batches")
print(f"[INFO] Valid dataloader: {len(valid_dataloader)} batches")
print("="*25 + "END STEP 1" + "="*25)

# Model
print(f"[PROGRESS] STEP 2: Initializing model...")
model = UNet()
model.apply(utils.weights_init)
moedl = nn.DataParallel(model)
model.to(device)

# Loss function
weights = torch.Tensor([[0.4, 0.55, 0.05]]).cuda()
loss_function = CEDiceLoss(weights)

# Optimizer
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
learing_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)

print("="*25 + "END STEP 2" + "="*25)

# Training
print(f"[PROGRESS] DTEP 3: Training...")
wandb.login(key = "09d9c01b499874346601e6ec425a8c58a6f82000")
wandb.init(project="PolypSegment")

for epoch in range(epochs):
    train_loss_epoch = 0
    test_loss_epoch = 0
    (train_loss_epoch, test_loss_epoch) = train.train(model= model, device= device, loss_function= loss_function, optimizer= optimizer, \
                                                      train_dataloader= train_dataloader, valid_dataloader= valid_dataloader, \
                                                      epoch= epoch, display_step= display_step, learing_rate_scheduler= learing_rate_scheduler)
    
    if test_loss_epoch < last_loss:
        utils.save_model(model, optimizer, checkpoint_path)
        last_loss = test_loss_epoch
        
    learing_rate_scheduler.step()
    train_loss_array.append(train_loss_epoch)
    test_loss_array.append(test_loss_epoch)
    wandb.log({"Train loss": train_loss_epoch, "Valid loss": test_loss_epoch})

print("="*25 + "END STEP 2" + "="*25)