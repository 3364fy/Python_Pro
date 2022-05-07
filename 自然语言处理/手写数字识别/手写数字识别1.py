import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# print(mnist[0][0])
# ret=transforms.ToTensor()(mnist[0][0])
# print(ret)
# print(ret.size())
#
# norm_img=transforms.Normalize((10),(1))(ret)
# print(norm_img)

BATCH_SIZE=128
TEST_BATCH_SIZE=1000
def get_dataloader(train=True,batch_size=BATCH_SIZE):
    dataset=MNIST(root='../data',train=True,download=False,transform=transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307),(0.3081))]))
    data_loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    return data_loader
# for i in get_dataloader():
#     print(i[0].size())


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1=nn.Linear(28*28*1,28)
        self.fc2=nn.Linear(28,10)

    def forward(self,x):
        x=x.view(-1,28*28*1)
        x=self.fc1(x)
        x=F.relu(x)
        out=self.fc2(x)
        return F.log_softmax(out,dim=-1)

mnist_net=MnistNet()
mnist_net.load_state_dict(torch.load('../data/model/mnist_net.pt'))
optimizer=Adam(mnist_net.parameters(),0.001)
optimizer.load_state_dict(torch.load('../data/model/mnist/mnist_optimizer.pt'))
def train(epoch):
    mode=True
    mnist_net.train(mode=mode)
    train_dataloader=get_dataloader(train=mode)
    for idx,(data,target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output=mnist_net(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if idx%10==0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss:{:.6f}'.format(
            #     epoch,idx*len(data),len(train_dataloader.dataset),100.*idx/len(train_dataloader),
            # loss.item()))
            print(epoch,idx,loss.item())

def test():
    loss_list=[]
    acc_list=[]
    mode = False
    test_dataloader = get_dataloader(train=mode,batch_size=TEST_BATCH_SIZE)
    for idx, (data, target) in enumerate(test_dataloader):
        with torch.no_grad():
            output=mnist_net(data)
            cur_loss=F.nll_loss(output,target)
            loss_list.append(cur_loss)
            pred=output.max(dim=1)[-1]
            cur_acc=pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print(f"平均准确率：{np.mean(acc_list)}\n平均损失：{np.mean(loss_list)}")

if __name__ == '__main__':
    # for i in range(30):
    #     train(i)
    # torch.save(mnist_net.state_dict(),'../data/model/mnist_net.pt')
    # torch.save(optimizer.state_dict(),'../data/model/mnist_optimizer.pt')
    test()




