import pandas as pd

import requests
from bs4 import BeautifulSoup
import json
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
}
#视频bv
# bv = 'https://www.bilibili.com/video/BV1RT4y1F7qn'
import re

df = pd.read_csv('D:/workingspace/Github/summary-notebooks-of-postgraduate/Python/bilibili_review/all_video_info.csv')
# print(df)

bv_list = df['bvid'].tolist()
# print(bv_list)

# bv_list=[]

def dec(x):
	r=0
	for i in range(6):
		r+=tr[x[s[i]]]*58**i
	return (r-add)^xor

def enc(x):
	x=(x^xor)+add
	r=list('BV1  4 1 7  ')
	for i in range(6):
		r[s[i]]=table[x//58**i%58]
	return ''.join(r)

fp = open('D:/workingspace/Github/summary-notebooks-of-postgraduate/Python/bilibili_review/'+'all_comments'+'.txt',"w",encoding="UTF-8")

for bv in bv_list:
    # bv = re.search(r'BV.+', bv).group()
    #评论页数
    pn = 1
    #排序种类 0是按时间排序 2是按热度排序
    sort = 2

    i=1
    panduan=0

    #bv，av互换算法
    table='fZodR9XQDSUm21yCkr6zBqiveYah8bt4xsWpHnJE7jL5VG3guMTKNPAwcF'
    tr={}
    for i in range(58):
        tr[table[i]]=i
    s=[11,10,3,8,4,6]
    xor=177451812
    add=8728348608

    #bv转换成oid
    oid = dec(bv)

    # fp = open('D:/workingspace/Github/summary-notebooks-of-postgraduate/Python/bilibili_review/'+bv+'.txt',"w",encoding="UTF-8")
    fp.write('*'*44+'\n')
    fp.write(str(bv)+'\n')
    fp.write('*'*44+'\n')
    while True:

        url =f'https://api.bilibili.com/x/v2/reply?pn={pn}&type=1&oid={oid}&sort={sort}'
        reponse = requests.get(url,headers=headers)
        a = json.loads(reponse.text)
        if pn==1:
            count = a['data']['page']['count']
            size = a['data']['page']['size']
            page = count//size+1
            print(page)
        try:
            for b in a['data']['replies']:
                panduan = 0
                str1=''
                str_list = list(b['content']['message'])
                for x in range(len(str_list)):
                    if str_list[x]=='[':
                        panduan=1
                    if panduan!=1:
                        str1 = str1+str_list[x]
                    if str_list[x] == ']':
                        panduan=0
                fp.write(str(i)+'、'+str1+'\n'+'-'*10+'\n')
                print(str1)
                print('-'*10)
                i = i + 1
        except:
            pass
        if pn!=page:
            pn += 1
        else:
            # fp.close()
            break
fp.close()