import zhconv

def hant_2_hans(hant_str: str):
    '''
    Function: 将 hant_str 由繁体转化为简体
    '''
    return zhconv.convert(hant_str, 'zh-hans')
a=hant_2_hans('誰教冥路作詩仙卿一愴然')
print(a)
