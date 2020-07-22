import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt 
from PIL import Image

import torch 
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F 
import torch.optim as optim 
import train_pytorch


# Load data
data_dir = '...'



# Load model
model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)).to(device)
model.load_state_dict(torch.load('exp/model/pytorch/weights.h5'))

# Evaluate on test images
validation_img_paths = ["validation/alien/11.jpg",
                        "validation/alien/22.jpg",
                        "validation/predator/33.jpg"]

img_list = [Image.open(input_path + img_path) for img_path in validation_img_paths]

validation_batch = torch.stack([data_transforms['validation'](img).to(device) \
    for img in img_list])

pred_logits_tensor = model(validation_batch)
pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

# Show results
fig, axs = plt.subplot(1, len(img_list), figsize = (20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    ax.axis('off')
    ax.imshow(img)

