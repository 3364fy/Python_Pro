import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.optim import SGD
x=torch.rand([20,1])
y_true=3*x**2+8

# x=torch.tensor([[48],[5],[28],[43],[88],[24]],dtype=torch.float)
# y_true=torch.tensor([[12],[64],[55],[16],[46],[37]],dtype=torch.float)
class Myliner(nn.Module):
    def __init__(self):
        super(Myliner,self).__init__()
        #传入特征数量，输出特征数量（传入列数，输出列数）
        self.linear=nn.Linear(1,100)
        self.relu=nn.ReLU()
        self.out=nn.Linear(100,1)
    #默认input，输入x
    def forward(self,x):
        out=self.linear(x)
        out=self.relu(out)
        out=self.out(out)
        return out

myliner=Myliner()
#最优化策略
optimizer=SGD(myliner.parameters(),0.001)
#损失函数
loss_fn=nn.MSELoss()

for i in range(40000):
    y_predict=myliner(x)
    loss=loss_fn(y_predict,y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%500==0:
        print(loss.item())
        print(list(myliner.parameters()))
        print(list(myliner.parameters())[0].size())
        print(list(myliner.parameters())[1].size())
        print(list(myliner.parameters())[2].size())

#设置模式为评估模式，即预测模式
myliner.eval()
predict=myliner(x)
predict=predict.data.numpy()
plt.scatter(x.data.numpy(),y_true.data.numpy(),c='r')
plt.scatter(x.data.numpy(),predict)
plt.show()


