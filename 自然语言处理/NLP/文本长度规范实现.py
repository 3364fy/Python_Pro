from keras.preprocessing import sequence
#cutlen根据数据分析中句子长度分布，覆盖90%左右预料的最短长度，这里假定cutlen为10
cutlen=10
def padding(x_train):
    return sequence.pad_sequences(x_train,cutlen)

x_train=[[1,23,55,32,48,78,63,42,8,77,52],[45,12,88,66,23]]
res=padding(x_train)
print(res)
