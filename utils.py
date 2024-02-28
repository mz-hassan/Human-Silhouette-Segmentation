import albumentations as A
from tqdm import tqdm

def get_train_augs(image_size):
  return A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p = 0.5),
    A.VerticalFlip(p = 0.5),
  ], is_check_shapes=False)

def get_val_augs(image_size):
  return A.Compose([
      A.Resize(image_size, image_size),
  ], is_check_shapes=False)

def train_fn(data_loader, model, optimizer, device="cuda"):
  model.train() # puts model to train model. activates dropout etc
  total_loss = 0.0

  for images, masks in tqdm(data_loader):
    images = images.to(device)
    masks = masks.to(device)

    optimizer.zero_grad()
    logits, loss = model(images, masks)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  return total_loss / len(data_loader)

def eval_fn(data_loader, model, device="cuda"):
  model.eval()
  total_loss = 0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader):
      images = images.to(device)
      masks = masks.to(device)

      logits, loss = model(images, masks)

      total_loss += loss.item()

  return total_loss / len(data_loader)