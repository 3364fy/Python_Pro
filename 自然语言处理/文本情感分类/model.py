import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from dataset import get_dataloader
from lib import ws, max_len


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        #词语的数量，维度
        self.embedding=nn.Embedding(len(ws),300)
        self.fc=nn.Linear(max_len*300,2)
    def forward(self,input):
        '''

        :param input: [batch_size,max_len]
        :return:
        '''
        x=self.embedding(input)#形状：[batch_size,max_len(行),100（列）]
        #print(f'x:{x}')

        out=self.fc(x.view([-1,max_len*300]))
        #print(f'out:{out}')
        #计算概率
        print(F.softmax(out, dim=-1))
        #print(F.log_softmax(out, dim=-1))
        # a=list(map(fy,F.log_softmax(out,dim=-1).max(dim=1)[-1]))
        # print(a)
        #print(F.log_softmax(out,dim=-1))
        #print('类别：',F.log_softmax(out,dim=-1).max(dim=1)[-1])
        return F.log_softmax(out,dim=-1)
def fy(a):
    if a==1:
        return '积极'
    else:
        return '消极'
model=Mymodel()
model.load_state_dict(torch.load('../data/model/情感分类/model.pt'))
optimizer=Adam(model.parameters(),0.01)
optimizer.load_state_dict(torch.load('../data/model/情感分类/model_optimizer.pt'))
def train(epoch):
    for idx,(input,target) in enumerate(get_dataloader(train=True)):
        # print(f'input:{input}')
        # print(f'target:{target}')
        optimizer.zero_grad()
        output=model.forward(input)
        #print(output)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if epoch%2==0:
            print('损失：',loss.item())
            # print(list(model.parameters())[0])
            # print(list(model.parameters())[0].size())
            # print(list(model.parameters())[1])
            # print(list(model.parameters())[1].size())
            # print(list(model.parameters())[2])
            # print(list(model.parameters())[2].size())


def test():
    loss_list=[]
    acc_list=[]
    mode = False
    model.eval()
    test_dataloader = get_dataloader(train=mode,batch=1000)
    for idx, (data, target) in enumerate(test_dataloader):
        with torch.no_grad():
            output=model(data)
            cur_loss=F.nll_loss(output,target)
            loss_list.append(cur_loss)
            pred=output.max(dim=1)[-1]
            cur_acc=pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print(f"平均准确率：{np.mean(acc_list)}\n平均损失：{np.mean(loss_list)}")

if __name__ == '__main__':
    for i in tqdm(range(1)):
        train(i)
    torch.save(model.state_dict(),'../data/model/情感分类/model.pt')
    torch.save(optimizer.state_dict(),'../data/model/情感分类/model_optimizer.pt')
    #test()