import os.path

import torch
from torch.utils.data import DataLoader, Dataset

from lib import sx, max_len, batch_size

train_data_path=r'E:\math\train'
test_data_path=r'E:\math\test'

def collate_fn(batch):
    content,label,title=list(zip(*batch))
    content=[sx.transform(i,max_len=max_len) for i in content]
    content=torch.LongTensor(content)
    label=torch.LongTensor(label)
    #label=torch.tensor([[label]])
    return content,label,title

class ImdbDataset(Dataset):
    def __init__(self,train=True):
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path
        data_path=self.train_data_path if train else self.test_data_path
        temp_data_path=[os.path.join(data_path,'唐诗'),os.path.join(data_path,'宋诗')]
        self.total_file_path=[]
        self.title=[]
        for path in temp_data_path:
            file_name_list=os.listdir(path)
            for i in file_name_list:
                self.title.append(i)
            file_path_list=[os.path.join(path,i) for i in file_name_list if i.endswith('.txt')]
            self.total_file_path.extend(file_path_list)
    def __getitem__(self, index):
        file_path=self.total_file_path[index]
        label_str=file_path.split('\\')[-2]
        label=1 if label_str=='唐诗' else 0
        tokens=open(file_path,encoding='utf-8').read().split(' ')
        return tokens,label,self.title[index].strip('.txt')
    def __len__(self):
        return len(self.total_file_path)

def get_dataloader(train=True,batch=batch_size):
    poet_dataset = ImdbDataset()
    data_loader = DataLoader(poet_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    for idx,(input,target,title) in enumerate(get_dataloader(train=True)):
        print(idx,input,target,title)
        break
    # a=torch.rand(2)
    # b = torch.rand(2,2)
    # print(a,b)
