import torch
from matplotlib import pyplot as plt

learning_rate=0.01
x=torch.rand([100,1])
#print(x,end='\n===================\n')
y_true=x*3+0.8
#print(y_true,end='\n===================\n')

w=torch.rand([1,1],requires_grad=True)
#print(w,end='\n===================\n')
b=torch.tensor(0.0,requires_grad=True)
#print(b,end='\n===================\n')


for i in range(10000):
    y_predict=torch.matmul(x,w)+b
    #print(y_predict)
    loss=(y_true-y_predict).pow(2).mean()
    #print(w.grad)

    if w.grad is not None:
        # print(w.grad)
        # print(w.grad.data)
        w.grad.data.zero_()

    if b.grad is not None:
        b.grad.data.zero_()

    loss.backward()
    w.data=w.data-learning_rate*w.grad
    b.data = b.data - learning_rate * b.grad
    print('w,b,loss',w.item(),b.item(),loss)

plt.figure(figsize=(20,20))
plt.scatter(x.numpy().reshape(-1),y_true.reshape(-1))
y_predict=torch.matmul(x,w)+b
plt.plot(x.numpy().reshape(-1),y_predict.detach().numpy().reshape(-1))
plt.show()