import requests 
import urllib  
import os

def post_login(url):
    print('\npost login')
    # http://pythonscraping.com/pages/cookies/login.html
    headers = {'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
                }

    payload = {'vToken': '','rdid': '1983060014', 'rdPasswd': 'Sdust@1983060014','returnUrl':'', 'password':''}
    r = requests.post(url, data=payload, headers = headers)
    print(r.cookies.get_dict())
    print(r.text)
    # http://pythonscraping.com/pages/cookies/profile.php
    # r = requests.get(url, headers = headers, cookies=r.cookies)
    # print(r.text)

def main():
    url = 'http://interlib.sdust.edu.cn/opac/reader/doLogin'
    # post_login(url)
    post_login(url)
    print('ok')    

if __name__=="__main__":
    main()
