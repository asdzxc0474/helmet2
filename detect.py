import torchvision.transforms as transforms
from torchvision import models
import torch
import torch.nn as nn
import torchvision
import cv2

class detect():
    def __init__(self):
        self.label=['with helmet','NO helmet' ]
        self.test_transform = transforms.Compose([transforms.Resize((224, 224)),torchvision.transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.feature_extract = True
        self.set_parameter_requires_grad(self.model, self.feature_extract)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, 2)
        self.model.load_state_dict(torch.load('./2saved_modelRes50.pt'))
        self.model= self.model.cuda()
        self.model.eval()
	
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def predict(self,frame):
        self.img = transforms.ToPILImage()(frame)
        self.img =self.test_transform(self.img)
        self.img = self.img.unsqueeze(0)
        self.outputs = self.model(self.img.cuda())
        #print(str(self.outputs.data))
        #_,self.predicted = torch.max(self.outputs.data,1)
        print(str(self.outputs.data[0][0].cpu().numpy()))
        if self.outputs.data[0][0].cpu().numpy()>=2:
            self.predicted = 0
        elif self.outputs.data[0][0].cpu().numpy()<=-0.3:
            self.predicted = 1
        else:
            self.predicted = self.predicted
        cv2.putText(frame, self.label[self.predicted],(50,80),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,255), 3, cv2.LINE_AA)
        return frame , self.predicted
