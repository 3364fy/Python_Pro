from torch.utils.data import Dataset,DataLoader
data_path=r'E:\python项目\自然语言处理\新闻分类\AG_NEWS\train.csv'

class MyDataset(Dataset):
    def __init__(self):
        self.lines=open(data_path).readlines()
    def __getitem__(self, index):
        #获取索引对应位置的一条数据
        cur_line=self.lines[index].strip()
        label=cur_line[1].strip()
        content=cur_line[3:].strip()
        return label,content
    def __len__(self):
        #返回数据总数量
        return len(self.lines)

my_dataset=MyDataset()
data_loader=DataLoader(dataset=my_dataset,batch_size=2,shuffle=True,num_workers=2,drop_last=True)

if __name__ == '__main__':

    # print(my_dataset[45])
    # print(len(my_dataset))
    # for i in data_loader:
    #     print(i)
    for index,(label,content) in enumerate(data_loader):
        print(index,label,content)
        print('*'*50)