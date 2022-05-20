class Word2Sequence():
    UNK_TAG='UNK'
    PAD_TAG='PAD'
    UNK=0
    PAD=1
    def __init__(self):
        self.dict={
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.count={}
    def fit(self,sentence):
        for word in sentence:
            self.count[word]=self.count.get(word,0)+1

    def bulid_vocab(self,min=None,max=None,max_futures=None):
        if min is not None:
            self.count={word:value for word,value in self.count.items() if value>min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value <max}
        if max_futures is not None:
            temp=sorted(self.count.items(),key=lambda x:x[-1],reverse=True)[:max_futures]
            self.count=dict(temp)
        for word in self.count:
            self.dict[word]=len(self.dict)
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentense,max_len=None):
        if max_len is not None:
            if max_len>len(sentense):
                sentense=sentense+[self.PAD_TAG]*(max_len-len(sentense))
            if max_len<len(sentense):
                sentense=sentense[:max_len]
        return [self.dict.get(word,self.UNK) for word in sentense]
    def inverse_transform(self,indices):
        return [self.inverse_dict.get(idx) for idx in indices]
    def __len__(self):
        #print(self.count)
        return len(self.dict)

if __name__ == '__main__':
    a=Word2Sequence()
    #a.fit(['我','喜','欢','你'])
    a.fit('不同的词语出现的次数不尽相同，是否需要对高频或者低频词语进行过滤，以及总的词语数量是否需要进行限制')
    print(a.count)

    a.bulid_vocab(max_futures=10)
    print(a.dict)
    b=a.transform('得到词典之后，如何把句子转化为数字序列，如何把数字序列转化为句子')
    print(b)
    c=a.inverse_transform([1,5,8,9])
    print(c)