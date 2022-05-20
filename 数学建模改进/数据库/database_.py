import pymysql
import pandas as pd
def f(a):
    return 1 if a=='唐' else 0
database = pymysql.connect(host="localhost", user="root", password="33649464", database="math", charset='utf8mb4')
cursor=database.cursor()
for i in range(64306,320858,8):
    #if i%4!=0:
        print(i)
        sql=f"select dynasty,title,split from poet where id='{i}'"
        cursor.execute(sql)
        result=cursor.fetchall()
        dynasty=result[0][0]
        title=result[0][1].replace('\t','')
        title.replace('/', '')
        split=result[0][2]
        with open(rf'E:\math\train\{dynasty}诗\{title}.txt','w',encoding='utf-8') as fp:
            fp.write(split)
    #else:
        print(i-4)
        sql = f"select dynasty,title,split from poet where id='{i-4}'"
        cursor.execute(sql)
        result = cursor.fetchall()
        dynasty = result[0][0]
        title = result[0][1].replace('\t','')
        title.replace('/','')
        split = result[0][2]
        with open(rf'E:\math\test\{dynasty}诗\{title}.txt', 'w', encoding='utf-8') as fp:
            fp.write(split)
