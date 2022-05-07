import pickle

ws=pickle.load(open('../data/model/情感分类/ws.pkl', 'rb'))

max_len=20
batch_size=256
#print(len(ws))