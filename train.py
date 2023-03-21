import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
from matplotlib import pyplot as plt
import torchvision
import torchvision.datasets as dsets

def imshow(img, title):
    img = torchvision.utils.make_grid(img, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

feature_extract = True
MAX_EPOCH= 300
BATCH_SIZE=64
LR=0.0001
num_classes = 2

train_dir="./target data/train"
val_dir = "./target data/val"
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = dsets.ImageFolder(train_dir, train_transform)
val_data = dsets.ImageFolder(val_dir, val_transform)

train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          pin_memory =True,
                          num_workers = 8)

valid_loader = DataLoader(val_data,
                          batch_size=16,
                          shuffle=True,
                          pin_memory =True,
                          num_workers = 8)

if __name__ == '__main__':
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    imshow(images, [train_data.classes[i] for i in labels])

    
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    #model.load_state_dict(torch.load('./2saved_modelRes50.pt'))

    if torch.cuda.is_available():
        model = model.cuda()

    # ============================ step 3/5 损失函数 ============================
    criterion=nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    optimizer=optim.Adam(model.parameters(),lr=LR, betas=(0.9, 0.99))# 选择优化器
    # ============================ step 5/5 训练 ============================
    # 记录每一次的数据，方便绘图
    min_valid_loss = np.inf 
    for e in range(MAX_EPOCH):
        train_loss = 0.0   # Optional when not using Model Specific layer
        model.train()
        for data, labels in train_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            target = model(data)
            loss = criterion(target,labels)
            a = loss.data.item()
            print('Epoch',e,'/',MAX_EPOCH,'train loss:{:.7f}'.format(a))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in valid_loader:
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            target = model(data)
            loss = criterion(target,labels)
            a = loss.data.item()
            print('Epoch',(e+1),'/',MAX_EPOCH,'val loss:{:.7f}'.format(a))
            valid_loss = loss.item() * data.size(0)

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.12f}--->{valid_loss:.12f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
        torch.save(model.state_dict(), '2saved_modelRes50.pt')
