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

BATCH_SIZE=64
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #定义第一层卷积神经网络,输入通道维度=1，输出通道维度=6，卷积核大小3*3
        self.layer1=nn.Sequential(nn.Linear(784,500),nn.Dropout(p=0.5),nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(500, 300), nn.Dropout(p=0.5), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(300, 10), nn.Softmax(dim=1))

    def forward(self,x):
        x=x.view(x.size()[0],-1)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

LR=0.5
model=Net()
mse_loss=nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(),LR)

def train():
    model.train()
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
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        _,predicted=torch.max(out,1)
        print(predicted == labels)
        correct+=(predicted==labels).sum()
        print(f'准确率：{correct/len(test_loader)}')
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        _,predicted=torch.max(out,1)
        print(predicted==labels)
        correct+=(predicted==labels).sum()
        print(f'准确率：{correct/len(train_loader)}')

for epoch in tqdm.tqdm(range(10)):
    print(f'epoch:{epoch}')
    train()
    test()