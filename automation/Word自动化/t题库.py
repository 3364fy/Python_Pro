import re

from docx import Document

#全是文字
document=Document('E:\安全管理信息系统.docx')
all_paragraphs=document.paragraphs
for paragraph in all_paragraphs:
    print(paragraph.text)
    ex='.+（ .+）.+'
    content=re.findall(ex,paragraph.text,re.S)
    print(content)