print('\033[0;34;41m\t\t图书音像勋章\033[m')
print('\033[0;31m\t\t==============================\033[m')
print('\033[0;32m\t\t==============================\033[m')
print('\033[0;33m\t\t==============================\033[m')
print('\033[0;35m\t\t==============================\033[m')
print('\033[0;36m\t\t==============================\033[m')
# import random
# while True:
#     a=random.randint(0,1)
#     b=random.randint(0,1)
#     print(f'\033[0;32m{a}\t{b}\033[m',end='')
#https://www.cnblogs.com/daofaziran/p/9015284.html
#\033[显示方式;前景色;背景色m + 结尾部分：\033[0m
'''显示方式: 
0（默认值）、1（高亮）、22（非粗体）、4（下划线）、24（非下划线）、 5（闪烁）、25（非闪烁）、7（反显）、27（非反显）
前景色: 30（黑色）、31（红色）、32（绿色）、 33（黄色）、34（蓝色）、35（洋 红）、36（青色）、37（白色）
背景色: 40（黑色）、41（红色）、42（绿色）、 43（黄色）、seckill（蓝色）、数学建模（洋 红）、46（青色）、47（白色）
常见开头格式：
\033[0m            默认字体正常显示，不高亮
\033[32;0m       红色字体正常显示
\033[1;32;40m  显示方式: 高亮    字体前景色：绿色  背景色：黑色
\033[0;31;46m  显示方式: 正常    字体前景色：红色  背景色：青色
\033[1;31m  显示方式: 高亮    字体前景色：红色  背景色：无'''