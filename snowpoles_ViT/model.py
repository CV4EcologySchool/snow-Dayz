import timm
import torch.nn as nn

def get_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    # Replace classifier head with regression head
    #model.head = nn.Linear(model.head.in_features, 1)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Replace and unfreeze the head
    model.head = nn.Linear(model.head.in_features, 1)
    for param in model.head.parameters():
        param.requires_grad = True

    return model