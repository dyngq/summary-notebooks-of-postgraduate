import urllib.request
import re
import os
import urllib


def get_html(url):
    page = urllib.request.urlopen(url)
    html_a = page.read()
    return html_a.decode('utf-8')


def get_urls(html):
    # reg = r'http://[^\s]*?\.jpg'
    reg = r'http://people.ucas.edu.cn/~[^\s]*?[^\"]*'
    urlreg = re.compile(reg)
    urls = urlreg.findall(html)  
    return urls

# names = ['yijianqiang','weiqinglai','wangshuo','liuzhiyong','houzengguang','tanmin','wangyunkuan','zhaoxiaoguang','chenglong','liuhongbin','wangyunkuan','wangjun','huangkaiqi','wangfeiyue','cengdajun','wangdanli','xiangshiming','heran','wangfeiyue','wangjun','xingjunliang','wangxingang','huweiming','tangming','tantieniu','wangjinqiao','xiangshiming','sunzhenan','zhangzhaoxiang','tangming','chengjian','xubo','zhouyu','zhangjiajun','liuyong','jiangtianzai','liubing','jiangtianzai','yangge','xubo','wangliang','zhangshuwu','wangpeng','wangdanli','wangjinqiao','luhanqing','zongchengqing','taojianhua','wangchunheng','sunzhenan','yangxin','chengjian','yangyiping','xingjunliang','dongqiulei','houzengguang','wangweiqun','wangpeng','zouwei','caozhiqiang','wangweiqun','wangshuo','caozhiqiang','tanmin','zouwei','wangxingang','fanguoliang','maowenji','liuchenglin','maowenji','zhangwensheng','lvpin','yangqing','huzhanyi','huangkaiqi','wangchunheng','liubing','libing','zhangxiaopeng','suijing','panchunhong','taojianhua','zongchengqing','liuzhenyu','fanlingzhong','liuyong','fanlingzhong','hanhua','cengyi','hehuiguang','zhangjiajun','qiaohong','zhangshuwu','liujing','libing','zhaojun','cengdajun','liuchenglin','huzhenhua','haojie','pengsilong','lvpin','panchunhong','chenglong','qiaohong','zhouchao','liangzize','fanguoliang','liangzize','xude','zhangzhengtao','yijianqiang','yangqing','liuzhiyong','yangyiping','zhaodongbin','wuyihong','guqingyi','dongqiulei','wangliang','liuzhenyu','yangge','guqingyi','pengsilong','haojie','zhaojun','hehuiguang','yangxin','yushan','huzhanyi','cengyi','zhangxiaopeng','wuyihong','xiaobaihua','huweiming','liukang','tantieniu','tanjie','xionggang','liuhongbin','zhaoxiaoguang','jingfengshui','zhouchao','jingfengshui','tanjie','zhaodongbin','zhangwensheng','zhangzhaoxiang','yushan','weiqinglai','zhangzhengtao','xude','heran','luhanqing','liujing','xiaobaihua','xuchangsheng','hanhua','liukang','huzhenhua','suijing','zhouyu','xuchangsheng','leizhen','xionggang','wangxuelei','liuzhen','bianguibin','tianbin','lien','hewenhao','xiexiaoliang','wubaolin','lilinjing','gaoyang','liqiudan','wangchuang','zhangyanming','wangjunping','zhanghaifeng','shenshuhan','menggaofeng','gaojin','wanjun','zhangyifan','liubin','heshizhu','wangkun','shangwenting','chenxi','wangwei','lvyanfeng','chewujun','chenzhineng','fangquan','dongweiming','zhangheng','maxiaojun','chenliang','liuyu','malei','wangjian','guodalei','wangshuangyi','liuxilong','yangguodong','zhengenhao','caiyinghao','shenfei','puzhiqiang','guanqiang','zhengxiaolong','yangxu','niwancheng','yinqiyue','lishuxiao','houxinwen','changhongxing','maxibo','cuiyue','xujiaming','zuonianming','dongdi','liguoqing','xingdengpeng','yandongming','yangminghao','wushu','dongjing','yinfei','yinzhigang','lixueen','wenglubin','lvyisheng','kangmengzhen','wangwei','sujianhua','wuzhengxing','zhengsuiwu','dengsai','zhangdapeng','xubo','yuanruyi','yuanyong','yangpeipei','sunzhengya','zhangjunge','wuhuaiyu','xushibiao','zhangfeng','wangwei','huochunlei','duyang','huihui','kongqingqun','mengweiliang','luoguan','zhangzhiwei','liaomingxue','zhufenghua','wangyu','jialihao','wuwei','lutao','lihaipeng','zhuyuanheng','caozhidong','zhangxuyao','shenzhen','wanglingfeng','gaowei','yangxiaoshan','songming','yangzhengyi','zhangqian','guojianwei','yuanchunfeng','hushaoyong','xuewenfang']

# "http://people.ucas.edu.cn/~haipengLee"

url = "http://www.ia.cas.cn/yjsjy/dsjj/"
# name = url.split('/')[-1][1:]
html_b = get_html(url) 
urls = get_urls(html_b)
print(urls)