# with open('E:/fil9','r',encoding='utf-8') as fp:
#     a=fp.read()
#     print(a[:80])

import fasttext
#使用fasttext的train_supervised（无监督训练方法）进行词向量的训练
#model=fasttext.train_supervised('fil9',dim=300,epoch=1,thread=12)
'cbow''skipgram'
# model=fasttext.train_supervised('fil9')
# model.save_model('fil9.bin')

# print(model.get_word_vector('the'))
# print(model.get_nearest_neighbors('sports'))
# print(model.get_nearest_neighbors('music'))
# print(model.get_nearest_neighbors('dog'))


model=fasttext.load_model('fil9.bin')
print(model.get_word_vector('the'))
print(model.get_nearest_neighbors('sport'))
print(model.get_nearest_neighbors('music'))
print(model.get_nearest_neighbors('dog'))