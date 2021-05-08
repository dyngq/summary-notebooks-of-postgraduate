title: Linux 权限详解
tags:
  - Linux
categories: Linux
date: 2020-11-08 08:45:00
---
# Linux 权限详解

## 基本原理

![img](https://img.dyngq.top/images/20201108212435.jpg)

| ALL  | 文件所有者 | 用户组 | 其它用户 |
| :--: | :--------: | :----: | :------: |
|  a   |     u      |   g    |    o     |
| all  |    user    | group  |  other   |

* 加权限 chmod u+rwx,g+rwx,o+rwx file
* 减权限 chmod u+rwx,g+rwx,o+rwx file
* a代表u+g+o，chmod a+rwx file
<!--more-->
* \+ 表示增加权限、- 表示取消权限、= 表示唯一设定权限。
* 其他参数
  * -c : 若该文件权限确实已经更改，才显示其更改动作
  * -f : 若该文件权限无法被更改也不要显示错误讯息
  * -v : 显示权限变更的详细资料
  * -R : 对目前目录下的所有文件与子目录进行相同的权限变更(即以递归的方式逐个变更)
  * --help : 显示辅助说明
  * --version : 显示版本

### 特殊权限

| rwx  | 读写执行     | ...                                                          |
| ---- | ------------ | ------------------------------------------------------------ |
| X    | 特殊执行权限 | 只有当文件为目录文件，或者其他类型的用户有可执行权限时，才将文件权限设置可执行 |
| s    | setuid/gid   | 当文件被执行时，根据who参数指定的用户类型设置文件的setuid或者setgid权限 |
| t    | 粘贴位       | 设置粘贴位，只有超级用户可以设置该位，只有文件所有者u可以使用该位 |

### 查看权限

> ls -la

* ![image-20201106173459491](https://img.dyngq.top/images/20201108212443.png)
* 第一位 d 代表文件夹
* ./ 代表当前目录
* ../代表父目录

## 八进制 快捷表示

> 根据 3位 二进制 来一一对应

|  #   |      权限      | rwx  | 二进制 |
| :--: | :------------: | :--: | :----: |
|  7   | 读 + 写 + 执行 | rwx  |  111   |
|  6   |    读 + 写     | rw-  |  110   |
|  5   |   读 + 执行    | r-x  |  101   |
|  4   |      只读      | r--  |  100   |
|  3   |   写 + 执行    | -wx  |  011   |
|  2   |      只写      | -w-  |  010   |
|  1   |     只执行     | --x  |  001   |
|  0   |       无       | ---  |  000   |

* 777 : rwxrwxrwx : ugo (a)
* 755 : rwx 

## 实际操作

* ![image-20201106170017841](https://img.dyngq.top/images/20201108212448.png)
* ![image-20201106170130163](https://img.dyngq.top/images/20201108212450.png)
* ![image-20201106170143787](https://img.dyngq.top/images/20201108212452.png)
* ![image-20201106170349767](https://img.dyngq.top/images/20201108212459.png)

## 参考链接

* [Linux chmod命令](https://www.runoob.com/linux/linux-comm-chmod.html)
* [inode-wiki](https://zh.wikipedia.org/wiki/Inode)