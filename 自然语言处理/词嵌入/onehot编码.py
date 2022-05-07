#导入用于对象保存与加载的joblib
import joblib
#词汇映射器
from keras.preprocessing.text import Tokenizer

def one_hot(vocab):
    #初始化一个词汇表
    # vocab={'周杰伦','陈奕迅','王力宏','李宗盛','吴亦凡','鹿晗'}
    #实例化一个词汇映射器
    t=Tokenizer(num_words=None,char_level=False)
    #在映射器上拟合现有的词汇表
    t.fit_on_texts(vocab)

    #循环遍历词汇表，将每一个单词映射为one-hot张量表示
    for token in vocab:
        #初始化一个全零的向量
        zero_list=[0]*len(vocab)
        #print(zero_list)
        #使用映射器转化现有的文本数据，每个词汇对应从开始的自然数
        #返回样式如[[2]],取出其中的数据需要使用[0][0]
        token_index=t.texts_to_sequences([token])[0][0]-1
        #print(t.texts_to_sequences([token]))
        #将对应的位置赋值为1
        zero_list[token_index]=1
        print(f'{token}的one-hot编码为：{zero_list}')
    #使用joblib工具保存映射，以便之后使用
    tokenizer_path='./Tokenizer'
    joblib.dump(t,tokenizer_path)

def hot_use(vocab):
    #加载之前保存的Tokenizer，实例化一个t对象
    t=joblib.load('Tokenizer')

    #编码token为'周杰伦'
    token='周杰伦'
    #使用t获得token_index
    token_index=t.texts_to_sequences([token])[0][0]-1
    #初始化一个zero_list
    zero_list=[0]*len(vocab)
    #令zero_list的对应索引为1
    zero_list[token_index]=1
    print(f'{token}的one-hot编码为：{zero_list}')
if __name__ == '__main__':
    vocab={'周杰伦','陈奕迅','王力宏','李宗盛','吴亦凡','鹿晗'}
    hot_use(vocab)