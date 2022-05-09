import torch

batch_size=10
seq_len=20
embedding_dim=30
word_vocab=100
hidden_size=18
num_layer=1
#准备输入数据
input=torch.randint(low=10,high=100,size=(batch_size,seq_len))
#print(input,end='\n'+'*'*100)

#准备embedding
embedding=torch.nn.Embedding(word_vocab,embedding_dim)
lstm=torch.nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=num_layer,
                   batch_first=True,bidirectional=True)

#进行embed操作
embed=embedding(input)#[10,20,30]
#print(embed,end='\n'+'*'*100)

#转换数据为batch_first=False
#embed=embed.permute(1,0,2)#[20,10,30]

#初始化状态，如果不初始化，torch默认初始值全为0
# h_0=torch.rand(num_layer,batch_size,hidden_size)
# c_0=torch.rand(num_layer,batch_size,hidden_size)
#output,(h_1,c_1)=lstm(embed,(h_0,c_0))
output,(h_1,c_1)=lstm(embed)
#output [20,10,1*18]
#h_1 [2,10,18]
#c_1 [2,10,18]

print(output,output.size(),end='\n'+'*'*100)
print(h_1,h_1.size(),end='\n'+'*'*100)
print(c_1,c_1.size(),end='\n'+'*'*100)

#获取最后一个时间步上的输入
last_output=output[:,-1,:]
#获得最后一次的hidden_state
last_hidden_state=h_1[-1,:,:]
print(last_output==last_hidden_state)
