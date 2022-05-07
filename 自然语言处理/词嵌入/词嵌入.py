import fileinput

import torch
from torch.utils.tensorboard import SummaryWriter

#tensorboard --logdir logs

#tensorboard --logdir runs --host localhost

#实例化一个摘要写入对象
writer=SummaryWriter()

#随机化一个100*50的矩阵，认为它是我们已经得到的词嵌入矩阵
#代表100个词汇，每个词汇被表示为50维的向量
embedded=torch.randn(100,50)

#导入事先准备好的100个中文词汇文件，形成meta列表原始词汇
meta=list(map(lambda x:x.strip(),fileinput.FileInput('vocab100.csv',encoding='utf-8')))
writer.add_embedding(embedded,metadata=meta)
writer.close()