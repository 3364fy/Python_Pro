#1. 对IMDB的数据记性fit操作
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImdbDataset
from dataset import tokenize

MAX_LEN=20
data_base_path = r"E:\aclImdb"
def fit_save_word_sequence():
    from wordSequence import Word2Sequence

    ws = Word2Sequence()
    train_path = [os.path.join(data_base_path,i)  for i in ["train/neg","train/pos"]]
    total_file_path_list = []
    for i in train_path:
        total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])
    for cur_path in tqdm(total_file_path_list,ascii=True,desc="fitting"):
        ws.fit(tokenize(open(cur_path,'r',encoding='utf-8').read().strip()))
    ws.build_vocab()
    # 对wordSequesnce进行保存
    pickle.dump(ws, open("../data/model/情感分类/ws2.pkl", "wb"))
    print(len(ws))

fit_save_word_sequence()

#2. 在dataset中使用wordsequence
ws = pickle.load(open("../data/model/情感分类/ws2.pkl", "rb"))

def collate_fn(batch):
    MAX_LEN = 500 
    #MAX_LEN = max([len(i) for i in texts]) #取当前batch的最大值作为batch的最大长度

    batch = list(zip(*batch))
    labes = torch.tensor(batch[0],dtype=torch.int)

    texts = batch[1]
    #获取每个文本的长度
    lengths = [len(i) if len(i)<MAX_LEN else MAX_LEN for i in texts]
    texts = torch.tensor([ws.transform(i, MAX_LEN) for i in texts])
    del batch
    return labes,texts,lengths
def get_dataloader(train=True):
    imdb_dataset = ImdbDataset(mode='train')
    data_loader = DataLoader(imdb_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    return data_loader
#3. 获取输出
dataset = ImdbDataset(ws=='train')
dataloader = DataLoader(dataset=dataset,batch_size=20,shuffle=True,collate_fn=collate_fn)
for idx,(label,text,length) in enumerate(dataloader):
    print("idx：",idx)
    print("table:",label)
    print("text:",text)
    print("length:",length)
    break