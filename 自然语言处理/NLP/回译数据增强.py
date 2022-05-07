from google_trans_new import google_translator
p_sample1='酒店设施非常不错'
p_sample2='这家价格很便宜'
n_sample1='拖鞋都发霉了，太差了'
n_sample2='电视机不好用，没有看到足球'
translator=google_translator()

translations = []
# 先进行中文到韩文的翻译
for text in [p_sample1, p_sample2, n_sample1, n_sample2]:
    translations.append(translator.translate(text, 'ko'))

# 获得翻译成韩文后的文本结果
ko_result = translations

print("中间翻译结果韩文是：",ko_result)

# 接下来进行回译
translations = translator.translate(ko_result, 'zh-cn')
cn_result = translations
print("回译得到的增强中文文本是：",cn_result)
