import os
import re

import torch
from torch.utils.data import DataLoader, Dataset

data_base_path = r"E:\aclImdb"


# 1. 定义tokenize的方法
def tokenize(text):
    # fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@'
        , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
    text = re.sub("<.*?>", " ", text, flags=re.S)
    text = re.sub("|".join(fileters), " ", text, flags=re.S)
    return [i.strip() for i in text.split()]


# 2. 准备dataset
class ImdbDataset(Dataset):
    def __init__(self, mode):
        super(ImdbDataset, self).__init__()
        if mode == "train":
            text_path = [os.path.join(data_base_path, i) for i in ["train/neg", "train/pos"]]
        else:
            text_path = [os.path.join(data_base_path, i) for i in ["test/neg", "test/pos"]]

        self.total_file_path_list = []
        for i in text_path:
            self.total_file_path_list.extend([os.path.join(i, j) for j in os.listdir(i)])

    def __getitem__(self, idx):
        cur_path = self.total_file_path_list[idx]

        cur_filename = os.path.basename(cur_path)
        label = int(cur_filename.split("_")[-1].split(".")[0]) - 1  # 处理标题，获取label，转化为从[0-9]
        text = tokenize(open(cur_path,encoding='utf-8').read().strip())  # 直接按照空格进行分词
        return label, text

    def __len__(self):
        return len(self.total_file_path_list)


# # 2. 实例化，准备dataloader
# dataset = ImdbDataset(mode="train")
# dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
#
# # 3. 观察数据输出结果
# for idx, (label, text) in enumerate(dataloader):
#     print("idx：", idx)
#     print("table:", label)
#     print("text:", text)
#     break

def collate_fn(batch):
    #batch是list，其中是一个一个元组，每个元组是dataset中__getitem__的结果
    batch = list(zip(*batch))
    labes = torch.tensor(batch[0],dtype=torch.int32)
    texts = batch[1]
    del batch
    return labes,texts
dataset = ImdbDataset(mode="train")
dataloader = DataLoader(dataset=dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)

if __name__ == '__main__':
    #此时输出正常
    for idx,(label,text) in enumerate(dataloader):
        print("idx：",idx)
        print("table:",label)
        print("text:",text)
        break