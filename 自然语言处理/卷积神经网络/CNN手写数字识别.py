import torch
import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

train_dataset=MNIST(root='../data',train=True,download=False,transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307),(0.3081))]))

test_dataset=MNIST(root='../data',train=False,download=False,transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307),(0.3081))]))

BATCH_SIZE=1024
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #定义第一层卷积神经网络,输入通道维度=1，输出通道维度=6，卷积核大小3*3
        self.conv1=nn.Sequential(nn.Conv2d(1,32,(5,5),(1,1),2),nn.ReLU(),nn.MaxPool2d(2,2))
        #池化之后图片大小变成（14,14）
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,(5,5),(1,1),2),nn.ReLU(),nn.MaxPool2d(2,2))
        # 池化之后图片大小变成（7,7）
        self.fc1 = nn.Sequential(nn.Linear(64*7*7, 1000), nn.Dropout(p=0.5),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1000, 10), nn.Softmax(dim=1))
    def forward(self,x):
        x=self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x=self.fc2(x)
        return x

LR=0.001
model=Net()
mse_loss=nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(),LR)

def train():
    model.train()
    print('训练集数量',len(train_dataset))
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out=model(inputs)
        loss=mse_loss(out,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def test():
    model.eval()
    correct=0
    print('测试集数量',len(test_dataset))
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        _,predicted=torch.max(out,1)
        print((predicted==labels).sum())
        correct+=(predicted==labels).sum()
    print(f'准确率：{correct/len(test_dataset)}')
    # for i, data in enumerate(train_loader):
    #     inputs, labels = data
    #     out = model(inputs)
    #     _,predicted=torch.max(out,1)
    #     correct+=(predicted==labels).sum()
    # print(f'准确率：{correct/len(train_loader)}')

for epoch in tqdm.tqdm(range(10)):
    print(f'epoch:{epoch}')
    train()
    test()