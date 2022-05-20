import os.path
import re

import torch
from torch.utils.data import DataLoader, Dataset

from lib import ws, max_len, batch_size

train_data_path=r'E:\aclImdb\train'
test_data_path=r'E:\aclImdb\test'

def tokenlie(content):
    content=re.sub('<.*?>','',content)
    filter=['\.','\t','\n','\x97','\x96','#','$','%','^','&']
    content=re.sub('|'.join(filter),'',content)
    tokens=[i.strip().lower() for i in content.split()]
    return tokens

def collate_fn(batch):
    content,label=list(zip(*batch))
    content=[ws.transform(i,max_len=max_len) for i in content]
    content=torch.LongTensor(content)
    label=torch.LongTensor(label)
    return content,label

class ImdbDataset(Dataset):
    def __init__(self,train=True):
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path
        data_path=self.train_data_path if train else self.test_data_path
        temp_data_path=[os.path.join(data_path,'pos'),os.path.join(data_path,'neg')]
        self.total_file_path=[]
        for path in temp_data_path:
            file_name_list=os.listdir(path)
            file_path_list=[os.path.join(path,i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)
    def __getitem__(self, index):
        file_path=self.total_file_path[index]
        label_str=file_path.split('\\')[-2]
        label=0 if label_str=='neg' else 1
        tokens=tokenlie(open(file_path,encoding='utf-8').read())
        return tokens,label
    def __len__(self):
        return len(self.total_file_path)

def get_dataloader(train=True,batch=batch_size ):
    imdb_dataset = ImdbDataset()
    data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    for idx,(input,target) in enumerate(get_dataloader(train=True)):
        print(idx,input,target)
        break
    # a=torch.rand(2)
    # b = torch.rand(2,2)
    # print(a,b)