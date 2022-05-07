#n-gram特征
#一般n-gram中n取2或者3，这里取2为例
ngram_range=2

def create_ngram_set(input_list):
    return set(zip(*[input_list[i:] for i in range(ngram_range)]))

input_list=[1,4,9,4,1,4]
res=create_ngram_set(input_list)
print(res)