import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.optim import SGD
x=torch.rand([500,1])
y_true=3*x+0.8

class Myliner(nn.Module):
    def __init__(self):
        super(Myliner,self).__init__()
        #传入特征数量，输出特征数量（传入列数，输出列数）
        self.linear=nn.Linear(1,1)
    #默认input，输入x
    def forward(self,x):
        out=self.linear(x)
        return out

myliner=Myliner()
#最优化策略
optimizer=SGD(myliner.parameters(),0.001)
#损失函数
loss_fn=nn.MSELoss()

for i in range(50000):
    y_predict=myliner(x)
    loss=loss_fn(y_predict,y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%500==0:
        print(loss.item(),list(myliner.parameters()))

#设置模式为评估模式，即预测模式
myliner.eval()
predict=myliner(x)
predict=predict.data.numpy()
plt.scatter(x.data.numpy(),y_true.data.numpy(),c='r')
plt.plot(x.data.numpy(),predict)
plt.show()


