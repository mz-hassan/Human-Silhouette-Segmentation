import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(nn.Module):
  def __init__(self, encoder, weights):
    super(SegmentationModel, self).__init__()
    self.arch = smp.Unet(
        encoder_name = encoder,
        encoder_weights = weights,
        in_channels = 3,
        classes = 1,
        activation = None # final ouputs as logits, not sigmoid
    )

  def forward(self, images, masks = None):
    logits = self.arch(images)

    if masks != None:
      loss1 = DiceLoss(mode='binary')(logits, masks)
      loss2 = nn.BCEWithLogitsLoss()(logits, masks)

      return logits, loss1+loss2 # for training

    return logits # for testing/inference