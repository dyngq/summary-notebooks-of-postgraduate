import urllib.request
import re
import os
import urllib

err = []

def get_html(url):
    page = urllib.request.urlopen(url)
    html_a = page.read()
    return html_a.decode('utf-8')


def get_img(html, name, num):
    # reg = r'http://[^\s]*?\.jpg'
    reg = r'/self/img/[^\s]*?.jpg'
    imgre = re.compile(reg)  # 转换成一个正则对象
    imglist = imgre.findall(html)  # 表示在整个网页过滤出所有图片的地址，放在imgList中
    # nums = 0        # 声明一个变量赋值
    path = './imgs/'  # 设置图片的保存地址
    if not os.path.isdir(path):
        os.makedirs(path)  # 判断没有此路径则创建
    paths = path  # 保存在test路径下
    global err
    for imgurl in imglist:
        # nums += 1
        x = name
        x = x.split('?')[0]
        # print('http://people.ucas.ac.cn' + imgurl)
        print('{0}{1}.jpg'.format(paths, x))
        try:
            urllib.request.urlretrieve('http://people.ucas.ac.cn' + imgurl, '{0}{1:02d}_{2}.jpg'.format(paths, num, x))  # 打开imgList,下载图片到本地
        except:
            print(x)
            err.append(x)
        # x = x + 1
        # print('图片开始下载，注意查看文件夹')
    return imglist

names = ['yijianqiang','weiqinglai','wangshuo','liuzhiyong','houzengguang','tanmin','wangyunkuan','zhaoxiaoguang','chenglong','liuhongbin','wangyunkuan','wangjun','huangkaiqi','wangfeiyue','cengdajun','wangdanli','xiangshiming','heran','wangfeiyue','wangjun','xingjunliang','wangxingang','huweiming','tangming','tantieniu','wangjinqiao','xiangshiming','sunzhenan','zhangzhaoxiang','tangming','chengjian','xubo','zhouyu','zhangjiajun','liuyong','jiangtianzai','liubing','jiangtianzai','yangge','xubo','wangliang','zhangshuwu','wangpeng','wangdanli','wangjinqiao','luhanqing','zongchengqing','taojianhua','wangchunheng','sunzhenan','yangxin','chengjian','yangyiping','xingjunliang','dongqiulei','houzengguang','wangweiqun','wangpeng','zouwei','caozhiqiang','wangweiqun','wangshuo','caozhiqiang','tanmin','zouwei','wangxingang','fanguoliang','maowenji','liuchenglin','maowenji','zhangwensheng','lvpin','yangqing','huzhanyi','huangkaiqi','wangchunheng','liubing','libing','zhangxiaopeng','suijing','panchunhong','taojianhua','zongchengqing','liuzhenyu','fanlingzhong','liuyong','fanlingzhong','hanhua','cengyi','hehuiguang','zhangjiajun','qiaohong','zhangshuwu','liujing','libing','zhaojun','cengdajun','liuchenglin','huzhenhua','haojie','pengsilong','lvpin','panchunhong','chenglong','qiaohong','zhouchao','liangzize','fanguoliang','liangzize','xude','zhangzhengtao','yijianqiang','yangqing','liuzhiyong','yangyiping','zhaodongbin','wuyihong','guqingyi','dongqiulei','wangliang','liuzhenyu','yangge','guqingyi','pengsilong','haojie','zhaojun','hehuiguang','yangxin','yushan','huzhanyi','cengyi','zhangxiaopeng','wuyihong','xiaobaihua','huweiming','liukang','tantieniu','tanjie','xionggang','liuhongbin','zhaoxiaoguang','jingfengshui','zhouchao','jingfengshui','tanjie','zhaodongbin','zhangwensheng','zhangzhaoxiang','yushan','weiqinglai','zhangzhengtao','xude','heran','luhanqing','liujing','xiaobaihua','xuchangsheng','hanhua','liukang','huzhenhua','suijing','zhouyu','xuchangsheng','leizhen','xionggang','wangxuelei','liuzhen','bianguibin','tianbin','lien','hewenhao','xiexiaoliang','wubaolin','lilinjing','gaoyang','liqiudan','wangchuang','zhangyanming','wangjunping','zhanghaifeng','shenshuhan','menggaofeng','gaojin','wanjun','zhangyifan','liubin','heshizhu','wangkun','shangwenting','chenxi','wangwei','lvyanfeng','chewujun','chenzhineng','fangquan','dongweiming','zhangheng','maxiaojun','chenliang','liuyu','malei','wangjian','guodalei','wangshuangyi','liuxilong','yangguodong','zhengenhao','caiyinghao','shenfei','puzhiqiang','guanqiang','zhengxiaolong','yangxu','niwancheng','yinqiyue','lishuxiao','houxinwen','changhongxing','maxibo','cuiyue','xujiaming','zuonianming','dongdi','liguoqing','xingdengpeng','yandongming','yangminghao','wushu','dongjing','yinfei','yinzhigang','lixueen','wenglubin','lvyisheng','kangmengzhen','wangwei','sujianhua','wuzhengxing','zhengsuiwu','dengsai','zhangdapeng','xubo','yuanruyi','yuanyong','yangpeipei','sunzhengya','zhangjunge','wuhuaiyu','xushibiao','zhangfeng','wangwei','huochunlei','duyang','huihui','kongqingqun','mengweiliang','luoguan','zhangzhiwei','liaomingxue','zhufenghua','wangyu','jialihao','wuwei','lutao','lihaipeng','zhuyuanheng','caozhidong','zhangxuyao','shenzhen','wanglingfeng','gaowei','yangxiaoshan','songming','yangzhengyi','zhangqian','guojianwei','yuanchunfeng','hushaoyong','xuewenfang']

urls = ['http://people.ucas.edu.cn/~lvyisheng', 'http://people.ucas.edu.cn/~liuzhen', 'http://people.ucas.edu.cn/~wangpengcasia', 'http://people.ucas.edu.cn/~liuhongbin', 'http://people.ucas.edu.cn/~bianguibin', 'http://people.ucas.edu.cn/~shuangyiwang', 'http://people.ucas.edu.cn/~wangwei', 'http://people.ucas.edu.cn/~wangyu_CASIA', 'http://people.ucas.edu.cn/~tianbin', 'http://people.ucas.edu.cn/~liuxilong', 'http://people.ucas.edu.cn/~jialihao', 'http://people.ucas.edu.cn/~yangguodong', 'http://people.ucas.edu.cn/~wei.wu', 'http://people.ucas.edu.cn/~zhengenhao', 'http://people.ucas.edu.cn/~suiwuzheng', 'http://people.ucas.edu.cn/~XieXiaoLiang', 'http://people.ucas.edu.cn/~yhcai', 'http://people.ucas.edu.cn/~dengsai', 'http://people.ucas.edu.cn/~haipengLee', 'http://people.ucas.edu.cn/~liuhongbin', 'http://people.ucas.edu.cn/~shenfei', 'http://people.ucas.edu.cn/~ljli', 'http://people.ucas.edu.cn/~pzq', 'http://people.ucas.edu.cn/~xubo_casia', 'http://people.ucas.edu.cn/~zhuyuanheng', 'http://people.ucas.edu.cn/~yang.gao', 'http://people.ucas.edu.cn/~guanqiang', 'http://people.ucas.edu.cn/~yuanruyi', 'http://people.ucas.edu.cn/~yangqing', 'http://people.ucas.edu.cn/~xlzheng', 'http://people.ucas.edu.cn/~yuanyong', 'http://people.ucas.edu.cn/~danliwang', 'http://people.ucas.edu.cn/~chuangwang', 'http://people.ucas.edu.cn/~XuYang', 'http://people.ucas.edu.cn/~ppyang', 'http://people.ucas.edu.cn/~xuyaozhang', 'http://people.ucas.edu.cn/~ymzhang', 'http://people.ucas.edu.cn/~junping.wang', 'http://people.ucas.edu.cn/~wanchengni', 'http://people.ucas.edu.cn/~szy', 'http://people.ucas.edu.cn/~zhenshen', 'http://people.ucas.edu.cn/~zhf', 'http://people.ucas.edu.cn/~yqy', 'http://people.ucas.edu.cn/~ZHANGJUNGE', 'http://people.ucas.edu.cn/~lvpin', 'http://people.ucas.edu.cn/~yangqing', 'http://people.ucas.edu.cn/~lfwang', 'http://people.ucas.edu.cn/~shibiaoxu', 'http://people.ucas.edu.cn/~mingt', 'http://people.ucas.edu.cn/~jgao', 
'http://people.ucas.edu.cn/~zfengia', 'http://people.ucas.edu.cn/~jwan', 'http://people.ucas.edu.cn/~xibo.ma', 'http://people.ucas.edu.cn/~wwong', 'http://people.ucas.edu.cn/~yangxiaoshan', 'http://people.ucas.edu.cn/~zhenyu', 'http://people.ucas.edu.cn/~yifanzhang', 'http://people.ucas.edu.cn/~cuiyue', 'http://people.ucas.edu.cn/~bingli', 'http://people.ucas.edu.cn/~geyang_ia', 'http://people.ucas.edu.cn/~mingt', 'http://people.ucas.edu.cn/~haojie', 'http://people.ucas.edu.cn/~bin.liu', 'http://people.ucas.edu.cn/~xujiaming', 'http://people.ucas.edu.cn/~shizhuhe', 'http://people.ucas.edu.cn/~zhenyu', 'http://people.ucas.edu.cn/~nmzuo', 'http://people.ucas.edu.cn/~msong', 'http://people.ucas.edu.cn/~yangxin', 'http://people.ucas.edu.cn/~shangwenting', 'http://people.ucas.edu.cn/~huihui', 'http://people.ucas.edu.cn/~zhengyiyang', 'http://people.ucas.edu.cn/~geyang_ia', 'http://people.ucas.edu.cn/~xichen', 'http://people.ucas.edu.cn/~guoqing.li', 'http://people.ucas.edu.cn/~wangwei_nlpr', 'http://people.ucas.edu.cn/~qqkong', 'http://people.ucas.edu.cn/~zhangqian', 'http://people.ucas.edu.cn/~lvyanfeng', 'http://people.ucas.edu.cn/~wangpengcasia', 'http://people.ucas.edu.cn/~danliwang', 'http://people.ucas.edu.cn/~cwj', 'http://people.ucas.edu.cn/~mengweiliang', 'http://people.ucas.edu.cn/~jianweiguo', 'http://people.ucas.edu.cn/~znchen', 'http://people.ucas.edu.cn/~yangminghao', 'http://people.ucas.edu.cn/~bingli', 'http://people.ucas.edu.cn/~quanfang', 'http://people.ucas.edu.cn/~shuwu', 'http://people.ucas.edu.cn/~luoguan?language=en', 'http://people.ucas.edu.cn/~yuanchunfeng', 'http://people.ucas.edu.cn/~dongjing', 'http://people.ucas.edu.cn/~hzhang2019', 'http://people.ucas.edu.cn/~fyin', 'http://people.ucas.edu.cn/~yangxin', 'http://people.ucas.edu.cn/~haojie', 'http://people.ucas.edu.cn/~nadectc', 'http://people.ucas.edu.cn/~zhangzw', 'http://people.ucas.edu.cn/~husy', 'http://people.ucas.edu.cn/~liangchen', 'http://people.ucas.edu.cn/~liaomingxue', 'http://people.ucas.edu.cn/~lvpin', 'http://people.ucas.edu.cn/~malei', 'http://people.ucas.edu.cn/~lbweng']

for num, url in enumerate(urls):
    # print(num, url)
    # url = "http://people.ucas.ac.cn/~" + name
    # print(url)
    # url = "http://people.ucas.ac.cn/~jlxing"
    name = url.split('/')[-1][1:]
    # print(name)
    html_b = get_html(url)  # 获取该网页的详细信息
    get_img(html_b, name, num)  # 从网页源代码中分析下载保存图片

