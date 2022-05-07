import json
import os

import jieba
import pymysql
import zhconv

database = pymysql.connect(host="localhost", user="root", password="33649464", database="math", charset='utf8mb4')
cursor = database.cursor()
def hant_2_hans(hant_str: str):
    '''
    Function: 将 hant_str 由繁体转化为简体
    '''
    return zhconv.convert(hant_str, 'zh-hans')
def poet(path,dynasty):
    files=os.listdir(path)
    id=64298
    for file in files:
        with open(f'{path}/{file}','r',encoding='utf-8') as fp:
            json_data=json.load(fp)
        for item in json_data:
            title=item['title']
            author=item['author']
            content_list=item['paragraphs']
            content=''.join(content_list)
            #sql = "insert into poet (id,dynasty,authors,contents) values ('%s','%s','%s','%s')" % (ids,dynastys,authors,contents)
            sql = f"insert into poet (id,dynasty,title,author,content) values ('{id}','{dynasty}','{title}','{author}','{content}')"
            cursor.execute(sql)
            database.commit()
            id+=1
    database.close()
def split():
    for i in range(1,320858):
        sql=f"select content from poet where id='{i}'"
        cursor.execute(sql)
        result=cursor.fetchall()[0][0]
        for junk in ["\n", " ", "。", "，", "[", "]", "《", "》", "（", "）", "〖", "〗", "『", "』", "：", "「", "」", "、", "{", "}",
                     "/", "=", "？", "1", "2", "3", "4", "5", "6", "7", "8", "9", ]:
            result=result.replace(f'{junk}','')
        split = jieba.lcut(result)
        #print(' '.join(split))
        sql = f"update poet set split ='{' '.join(split)}' where id ='{i}'"
        cursor.execute(sql)
        database.commit()
        i += 1
    database.close()
def font():
    for i in range(1, 320858):
        sql = f"select title,author,content,split from poet where id='{i}'"
        cursor.execute(sql)
        result = cursor.fetchall()
        dict={'title':0,'author':1,'content':2,'split':3}
        for field in dict:
            sql = f"update poet set {field} ='{hant_2_hans(result[0][dict[field]])}' where id ='{i}'"
            cursor.execute(sql)
            database.commit()
        i += 1
if __name__ == '__main__':
    split()

