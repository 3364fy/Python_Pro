import numpy as np

class Word2Sequence():
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG :self.UNK,
            self.PAD_TAG :self.PAD
        }
        self.fited = False

    def to_index(self,word):
        """word -> index"""
        assert self.fited == True,"必须先进行fit操作"
        return self.dict.get(word,self.UNK)

    def to_word(self,index):
        """index -> word"""
        assert self.fited , "必须先进行fit操作"
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    def __len__(self):
        return len(self.dict)

    def fit(self, sentences, min_count=1, max_count=None, max_feature=None):
        """
        :param sentences:[[word1,word2,word3],[word1,word3,wordn..],...]
        :param min_count: 最小出现的次数
        :param max_count: 最大出现的次数
        :param max_feature: 总词语的最大数量
        :return:
        """
        count = {}
        for sentence in sentences:
            for a in sentence:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        # 比最小的数量大和比最大的数量小的需要
        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}
        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        # 限制最大的数量
        if isinstance(max_feature, int):
            count = sorted(list(count.items()), key=lambda x: x[1])
            if max_feature is not None and len(count) > max_feature:
                count = count[-int(max_feature):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)

        self.fited = True
        # 准备一个index->word的字典
        self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence,max_len=None):
        """
        实现吧句子转化为数组（向量）
        :param sentence:
        :param max_len:
        :return:
        """
        assert self.fited, "必须先进行fit操作"
        if max_len is not None:
            r = [self.PAD]*max_len
        else:
            r = [self.PAD]*len(sentence)
        if max_len is not None and len(sentence)>max_len:
            sentence=sentence[:max_len]
        for index,word in enumerate(sentence):
            r[index] = self.to_index(word)
        return np.array(r,dtype=np.int64)

    def inverse_transform(self,indices):
        """
        实现从数组 转化为文字
        :param indices: [1,2,3....]
        :return:[word1,word2.....]
        """
        sentence = []
        for i in indices:
            word = self.to_word(i)
            sentence.append(word)
        return sentence

    def build_vocab(self):
        pass


if __name__ == '__main__':
    w2s = Word2Sequence()
    w2s.fit([
        ["你", "好", "么"],
        ["你", "好", "哦"]])

    print(w2s.dict)
    print(w2s.fited)
    print(w2s.transform(["你","好","嘛"]))
    print(w2s.transform(["你好嘛"],max_len=10))