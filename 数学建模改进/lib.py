import pickle

import torch

sx=pickle.load(open('data/BP/sx.pkl', 'rb'))

max_len=100
batch_size=1024
#56985
hidden_size=128
num_layers=2
bidirectional=True
dropout=0.4

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#print(len(sx))


