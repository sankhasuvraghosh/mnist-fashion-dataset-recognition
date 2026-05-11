import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as opt
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

device ="cuda" if torch.cuda.is_available() else "cpu"
print(f"using device : {device}")
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']
transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
train_data=FashionMNIST(root="data", download=True,train=True,transform=transform)
test_data=FashionMNIST(root="data",download=True,train=True,transform=transform)
train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
class fashion_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2,2))
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,2))
        self.conv3=nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.classifier=nn.Sequential(nn.Flatten(),
                                      nn.Linear(128*7*7,256),nn.ReLU(),nn.Dropout(p=0.5),
                                      nn.Linear(256,128),nn.ReLU(),nn.Dropout(p=0.3),
                                      nn.Linear(128,10))
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.classifier(x)
        return x
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
loss_fn=nn.CrossEntropyLoss()
model=fashion_model().to(device)
optimizer=opt.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)
scheduler=opt.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)
epochs=10
for epoch in range(epochs):
    model.train()
    total_loss=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        output=model(images)
        loss=loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    scheduler.step()
    print(f"epoch{epoch+1}/{epochs}    |   loss= {total_loss}")

model.eval()
correct=0
total=0
with torch.no_grad():
        for images,labels in test_loader:
         images=images.to(device)
         labels=labels.to(device)
         outputs=model(images)
         predictions=outputs.argmax(dim=1)
         correct+=(predictions==labels).sum().item()
         total+=labels.size(0)
accuracy=100*correct/total
print(f"accuracy : {accuracy}")
torch.save(model.state_dict(),"fashion_dataset_model.pth")
print("model saved as  : fashion_dataset_model.pth")
index=int(input("enter the index number of the fashion "))
image,true_label=test_data[index]
plt.imshow(image.squeeze(),cmap="gray")
plt.title(f"actual label : {class_names[true_label]}")
plt.axis("off")
plt.show()
image_flat=image.unsqueeze(0).to(device)
with torch.no_grad():
    output=model(image_flat)
    predicted_label=output.argmax(dim=1).item()
print("user picked image index", index)
print("actual label" ,class_names[true_label])
print("model prediction ", class_names[predicted_label])

