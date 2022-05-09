import pickle

import torch

ws=pickle.load(open('../../data/model/情感分类/ws.pkl', 'rb'))

max_len=200
batch_size=1024
hidden_size=128
num_layers=2
bidirectional=True
dropout=0.4

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(len(ws))