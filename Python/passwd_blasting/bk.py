from urllib import request,parse
from html.parser import HTMLParser
import urllib

global flag

def logon(testID,IDcard):
    PostUrl = 'http://*.*.*.*:*/zskc/checklogin'
    testID = testID

#构建登录data
    login_data = parse.urlencode([
    ('bkType', 'now'),
    ('rdid', testID),
    ('rdPasswd', IDcard),
    ])
    req = request.Request(PostUrl)

#构建登录head 请求头
    req.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'),
    req.add_header('Accept-Encoding', 'gzip, deflate'),
    req.add_header('Accept-Language', 'zh-CN,zh;q=0.9,en;q=0.8'),
    req.add_header('Cache-Control', 'max-age=0'),
    req.add_header('Connection', 'keep-alive'),
    req.add_header('Content-Length', '55'),
    req.add_header('Content-Type', 'application/x-www-form-urlencoded'),
    req.add_header('Cookie', 'JSESSIONID=1A5F7DD34539699E9EC7CB7298745713'),
    req.add_header('Host', '*.*.*.*:*'),
    req.add_header('Origin', 'http://*.*.*.*:*'),
    req.add_header('Referer', 'http://*.*.*.*:*/zskc/'),
    req.add_header('Upgrade-Insecure-Requests', '1'),
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36'),

    with request.urlopen(req,data=login_data.encode('utf-8')) as f:
        if f.read().decode('GBK').find('alert')==-1:
            global flag
            flag = 1
            print(f.read().decode('GBK'))
            print("准考证号为："+testID)
        return(f.read().decode('GBK'))

def main():
    idcard = '*************'
    for i in range(1075, 1500):
        global flag
        flag = 0
        a = '10424953000'
        a = a + str('%04d' % i)

        logon(a,idcard)
        del a
        if flag == 1:
            break

if __name__ == '__main__':
    main()