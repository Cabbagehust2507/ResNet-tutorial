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

input_path = "...."

# Create Pytorch data generators

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std= [0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
}

image_datasets = {
    'train':
    datasets.ImageFolder(input_path + 'train', data_transforms['train']),
    'validation':
    datasets.ImageFolder(input_path + 'val', data_transforms['validation'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=0),
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=16,
                                shuffle=False,
                                num_workers=0)
}

# Create the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2).to(device)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# Train the model

def train_model(model, criterion, optimizer, num_epochs = 3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets([phase]))

            print('{} loss: {:.4f}, acc: {:4f}'.format(phase, epoch_loss, epoch_acc))
    return model

model_trained = train_model(model, criterion, optimizer, num_epochs=3)

# Save the model

torch.save(model_trained.state_dict(), 'exp/model/pytorch/weights.h5')

