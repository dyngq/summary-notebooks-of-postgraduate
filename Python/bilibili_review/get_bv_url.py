import requests
from bs4 import BeautifulSoup
import json
import re

import pandas as pd
import csv

headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
}

uid = 43726449
page_nums = 3
with open('D:/workingspace/Github/summary-notebooks-of-postgraduate/Python/bilibili_review/'+'all_vedio_info'+'.txt',"w",encoding="UTF-8") as fp:
    
    fp.write("bvid"+', '+"description"+', '+"title"+'\n')
    for page_num in range(page_nums):
        page_num = page_num + 1
        url =f'https://api.bilibili.com/x/space/arc/search?mid={uid}&ps=30&tid=0&pn={page_num}&keyword=&order=pubdate&jsonp=jsonp'
        reponse = requests.get(url,headers=headers)
        a = json.loads(reponse.text)
        # print(reponse.text)
        # print(a['data']['page']['ps'])
        # print(a['data']['list']['vlist'])

        count_ = a['data']['page']['ps']
        # print(count_)
        
        for i in range(count_):
            try:
                bvid = a['data']['list']['vlist'][i]['bvid']
                description = a['data']['list']['vlist'][i]['description']
                title = a['data']['list']['vlist'][i]['title']
                print('--------------------------------------------------')
                print(bvid)                
                description = re.sub(r'\"',' ',description)
                description = re.sub(r'\n',' ',description)
                description =re.sub(r'吨吨吨吨吨，吨到世界充满爱！','-',description)
                print(description)
                print(title)
                fp.write(bvid+', '+description+', '+title+'\n')
            except:
                pass

# with open('D:/workingspace/Github/summary-notebooks-of-postgraduate/Python/bilibili_review/'+'all_video_info'+'.csv',"w",encoding="UTF-8") as csvfile:
    
#     writer = csv.writer(csvfile)
#     writer.writerow(["bvid","description","title"])

#     for page_num in range(page_nums):
#         page_num = page_num + 1
#         url =f'https://api.bilibili.com/x/space/arc/search?mid={uid}&ps=30&tid=0&pn={page_num}&keyword=&order=pubdate&jsonp=jsonp'
#         reponse = requests.get(url,headers=headers)
#         a = json.loads(reponse.text)
#         # print(reponse.text)
#         # print(a['data']['page']['ps'])
#         # print(a['data']['list']['vlist'])

#         count_ = a['data']['page']['ps']
#         # print(count_)
        
#         for i in range(count_):
#             try:
#                 bvid = a['data']['list']['vlist'][i]['bvid']
#                 description = a['data']['list']['vlist'][i]['description']
#                 title = a['data']['list']['vlist'][i]['title']
#                 print('--------------------------------------------------')
#                 print(bvid)                
#                 description = re.sub(r'\"',' ',description)
#                 description = re.sub(r'\n',' ',description)
#                 description =re.sub(r'吨吨吨吨吨，吨到世界充满爱！','-',description)
#                 print(description)
#                 print(title)
#                 writer.writerow([bvid,description,title])
#             except:
#                 pass