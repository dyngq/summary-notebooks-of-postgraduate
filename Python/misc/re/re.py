# import re

# list_re = ['a','ab','abb','aabb','abbb']
# for i in list_re:
#     print(re.search('^ab*$',i)) # .span从起始位置开始匹配

# print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
# print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配
# print(re.search('www', 'www.runoob.com').span())  # 在起始位置匹配
# print(re.search('com', 'www.runoob.com'))         # 不在起始位置匹配

# mm="c:\ab\bc\cd\\"
# print (mm)

# print(re.findall(r"abc","adsssa abc "))

#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import re
 
phone = "2004-959-559 # 这是一个国外电话号码"
 
# 删除字符串中的 Python注释 
num = re.sub(r'[^0-9|-]', '',phone)
print("电话号码是: ", num)

# 删除字符串中的 Python注释   2
num = re.sub(r'#.*$', '',phone)
print("电话号码是: ", num)

# 删除非数字(-)的字符串 
num = re.sub(r'[^0-9]', '',phone)
print("电话号码是 : ", num)

import re

s = "Today is 3/2/2017 。Pycon starts 5/25/2017"

new_s = re.sub("(\d+)/(\d+)/(\d+)",r"\3-\1-\2",s)
print(new_s)

new_s = re.sub("(\d+)/(\d+)/(\d+)","$3 $1 $2",s)
print(new_s)

def hello(**kwargs):
    print(kwargs)
    m = kwargs.pop("sex","nv")
    print(m , kwargs)

hello(name='zhaojinye' ,sex = "nv",husband = "yu")