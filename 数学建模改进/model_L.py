import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from dataset import get_dataloader
from lib import sx, hidden_size, num_layers, bidirectional, dropout, device


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        #词语的数量，维度
        self.embedding=nn.Embedding(len(sx),100)
        self.lstm=nn.LSTM(input_size=100,hidden_size=hidden_size,num_layers=num_layers,
                batch_first=True,bidirectional=bidirectional,dropout=dropout)
        self.fc=nn.Linear(hidden_size*2,2)
    def forward(self,input):
        '''

        :param input: [batch_size,max_len]
        :return:
        '''
        x=self.embedding(input)#形状：[batch_size,max_len(行),100（列）]
        #print(f'x:{x}')

        #x [batch_size,max_len,num_layers*hidden_size],h_n [2*num_layers,batch_size,hidden_size]
        x,(h_n,c_n)=self.lstm(x)

        #获取两个方向最后一次的output，进行concat
        output_fw=h_n[-2,:,:]#正向最后一次的输出
        output_bw=h_n[-1,:,:]#反向最后一次的输出
        output=torch.concat([output_fw,output_bw],dim=-1)#[batch_size,hidden_size*2]

        out=self.fc(output)
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
        return '唐诗'
    else:
        return '宋诗'
model=Mymodel().to(device)
#model.load_state_dict(torch.load('../../data/model/情感分类/model.pt'))
optimizer=Adam(model.parameters(),0.01)
#optimizer.load_state_dict(torch.load('../../data/model/情感分类/model_optimizer.pt'))
def train(epoch):
    for idx,(input,target,title) in enumerate(get_dataloader(train=True)):
        # print(f'input:{input}')
        # print(f'target:{target}')
        input=input.to(device)
        target=target.to(device)
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
    for idx, (data, target,title) in enumerate(test_dataloader):
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
        test()