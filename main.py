import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_train_augs, get_val_augs, train_fn, eval_fn
from dataset import SegmentationDataset
from model import SegmentationModel
from torch.utils.data import DataLoader

def main():
    CSV_FILE = 'Human-Segmentation-Dataset-master/train.csv'
    DATA_DIR = ''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 25 
    lr = 0.003
    image_size = 320
    batch_size = 16

    encoder = 'timm-efficientnet-b0'
    weights = 'imagenet'
    df = pd.read_csv(CSV_FILE)

    train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)

    trainset = SegmentationDataset(train_df, get_train_augs(image_size))
    validset = SegmentationDataset(valid_df, get_val_augs(image_size))

    print(f"Size of Trainset : {len(trainset)}")
    print(f"Size of Validset : {len(validset)}")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True)
    print(f"# Batches in train: {len(trainloader)}, valid: {len(validloader)}")

    for image, mask in trainloader:
        break
    print(f"One batch image shape: {image.shape}, mask: {mask.shape}")

    model = SegmentationModel(encoder, weights)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_validation_loss = np.Inf

    for i in range(epochs):
        train_loss = train_fn(trainloader, model, optimizer)
        valid_loss = eval_fn(validloader, model)

        if valid_loss < best_validation_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            print("Model saved")
            best_validation_loss = valid_loss

        print(f"Epoch {i+1} train loss: {train_loss}, valid loss: {valid_loss}")

if __name__ == '__main__':
    main()