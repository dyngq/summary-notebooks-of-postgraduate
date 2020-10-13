# python 小项目和tips等

## 小项目

* [准考证号“爆破”——解决忘记准考证号无法查询考研初始成绩（校官网）](https://github.com/dyngq/summary-notebooks-of-postgraduate/tree/master/Python/burst_number)
* 小脚本：批处理提取word信息到excel

## 数据类型

* `list`
* `tupple` 不可变
* `set`
  * `set()` 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
* `dict`

## 常用函数

* 时间序列 `datetime` （pandas也有）另外还有`time` `calender` （Py数分 P303）
  1. 两个datetime相减，产生`datetime.timedelta`类型的时间差数据，可以用来倍加倍减
  2. `time`(h s m 微秒) `date`(年月日) `datetime` `timedelta` `tzinfo`(时区)
  3. 字符串与datetime转换: 
     1. `str.strftime('%Y-%m-%d')`
     2. `datetime.strptime(str, '%m/%d/%Y')`
     3. 第三方包 from dateutil.parser import parse
     4. pandas.to_datetime([str])
* 

## tips

* 理解python**装饰器** 
  * python利用装饰器可以实现AOP面向切面编程，类似于java的注解@
  * 理解 闭包 
    * [什么是闭包？ - Saviio的回答 - 知乎](https://www.zhihu.com/question/34210214/answer/94933160)
    * [什么是闭包？ - 胖君的回答 - 知乎](https://www.zhihu.com/question/34210214/answer/110177125)
    * [为什么javascript closure(闭包)要叫闭包？](https://www.zhihu.com/question/35177512)
  * 什么是 AOP 什么是OOP
    * [AOP与OOP有什么区别，谈谈AOP的原理是什么](https://juejin.im/post/6844903961955139598)
    * [什么是面向切面编程AOP？ - 欲眼熊猫的回答 - 知乎](https://www.zhihu.com/question/24863332/answer/48376158)

## 一些python小知识积累

* 深拷贝与浅拷贝
* Python中*args和**kwargs的区别 ： [参考链接](https://www.cnblogs.com/yunguoxiaoqiao/p/7626992.html)
* [python numpy 常用随机数的产生方法](https://blog.csdn.net/m0_37804518/article/details/78490709)
* [ln -s 软链接知识总结](https://www.cnblogs.com/hxy5/p/9460063.html)

* python3——“->”的含义 ： ->用于指示函数返回的类型
!['dyngq_images'](images/dyngq_2020-03-20-00-13-45.png)

* **if _name_ == '_main_'** : 的意思是：当.py文件被直接运行时，if _name_ == '_main_'之下的代码块将被运行；当.py文件以模块形式被导入时，if _name_ == '_main_'之下的代码块不被运行。
* self.__classs__.__name__ 获取类名。
* getattr() 函数用于返回一个对象属性值。
* **dictionary.keys()** 方法是Python的字典方法，它将字典中的所有键组成一个可迭代序列并返回。在Python3中，keys函数不再返回一个列表，而是一个dict_keys类型的**可迭代序列**。

* [正则表达式](/misc/re/)
