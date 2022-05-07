import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.optim import SGD

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x=torch.rand([500,1]).to(device)
y_true=3*x+0.8


class Myliner(nn.Module):
    def __init__(self):
        super(Myliner,self).__init__()
        self.linear=nn.Linear(1,1)

    def forward(self,x):
        out=self.linear(x)
        return out

myliner=Myliner().to(device)
optimizer=SGD(myliner.parameters(),0.001)
loss_fn=nn.MSELoss()

for i in range(50000):
    y_predict=myliner(x)
    loss=loss_fn(y_predict,y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%50==0:
        print(loss.item(),list(myliner.parameters()))

#设置模式为评估模式，即预测模式
myliner.eval()
predict=myliner(x)
predict=predict.data.numpy()
plt.scatter(x.data.numpy(),y_true.data.numpy(),c='r')
plt.plot(x.data.numpy(),predict)
plt.show()



