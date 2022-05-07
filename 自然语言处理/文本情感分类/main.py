import os
import pickle

from tqdm import tqdm

from dataset import tokenlie
from word_sequence import Word2Sequence

if __name__ == '__main__':
    ws=Word2Sequence()
    path=r'E:\aclImdb\train'
    temp_data_path=[os.path.join(path,'pos'),os.path.join(path,'neg')]
    for data_path in temp_data_path:
        file_paths=[os.path.join(data_path,file_name) for file_name in os.listdir(data_path) if file_name.endswith('txt')]
        for file_path in tqdm(file_paths):
            sentence=tokenlie(open(file_path,encoding='utf-8').read())
            ws.fit(sentence)
    ws.bulid_vocab(min=10,max_futures=10000)
    pickle.dump(ws, open('../data/model/情感分类/ws.pkl', 'wb'))
    print(len(ws))