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

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        #input_size：输入特征的大小
        #hiddensize:LSTM模块的数量
        #numlayer:LSTM的层数
        #batch_first:LSTM默认输入格式input[seq_len(序列长度)，batch(批次大小),feature(特征数量)]
        #batch_first=True，input和output变为[batch(批次大小),seq_len(序列长度)，feature(特征数量)]
        self.lstm=nn.LSTM(input_size=28,hidden_size=64,num_layers=1,batch_first=True)
        self.out=nn.Linear(64,10)
        self.softmax=nn.Softmax(dim=1)


    def forward(self,x):
        x=x.view(-1,28,28)
        output,(h_n,h_c)=self.lstm(x)
        output_in_last_timestep=h_n[-1,:,:]
        x=self.out(output_in_last_timestep)
        x=self.softmax(x)
        return x

LR=0.001
model=LSTM()
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