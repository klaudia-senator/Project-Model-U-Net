import torch, torch.nn.functional as F

def dice_loss(logits, targets, smooth=1e-6):
    probs   = F.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(targets, probs.size(1)).permute(0,3,1,2).float()
    inter   = (probs * one_hot).sum((0,2,3))
    union   = (probs + one_hot).sum((0,2,3))
    dice    = (2*inter + smooth) / (union + smooth)
    return 1 - dice.mean()