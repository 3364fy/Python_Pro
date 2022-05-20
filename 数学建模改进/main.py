import os
import pickle

from tqdm import tqdm

from word_sequence import Word2Sequence

if __name__ == '__main__':
    sx=Word2Sequence()
    path=r'E:\math\train'
    temp_data_path=[os.path.join(path,'唐诗'),os.path.join(path,'宋诗')]
    for data_path in temp_data_path:
        file_paths=[os.path.join(data_path,file_name) for file_name in os.listdir(data_path) if file_name.endswith('txt')]
        for file_path in tqdm(file_paths):
            sentence=open(file_path,encoding='utf-8').read().split(' ')
            #print(sentence)
            sx.fit(sentence)
    sx.bulid_vocab(min=50,max_futures=10000)
    pickle.dump(sx, open('data/BP/sx.pkl', 'wb'))
    print(len(sx))