from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

class SegmentationDataset(Dataset):
  def __init__(self, df, augmentation):
    self.df = df
    self.augmentations = augmentation

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #since cv reads image in BGR

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255.0
    mask = np.expand_dims(mask, axis = -1) # add channel dim at end

    # print(image.shape, mask.shape)
    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']

    #(h,w,c) to (c,h,w)
    image  = np.transpose(image, (2,0,1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask)) #no need to scale alread [0,1]

    return image, mask