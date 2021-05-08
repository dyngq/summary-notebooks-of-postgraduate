# import time
# import os

# def long_time_task():
#     print('当前进程: {}'.format(os.getpid()))
#     time.sleep(2)
#     print("结果: {}".format(8 ** 20))

# if __name__ == "__main__":
#     print('当前母进程: {}'.format(os.getpid()))
#     start = time.time()
#     for i in range(2):
#         long_time_task()
#     end = time.time()
#     print("用时{}秒".format((end-start)))


# from multiprocessing import Process
# import os
# import time

# def long_time_task(i):
#     print('子进程: {} - 任务{}'.format(os.getpid(), i))
#     time.sleep(2)
#     print("结果: {}".format(8 ** 20))

# if __name__=='__main__':
#     print('当前母进程: {}'.format(os.getpid()))
#     start = time.time()
#     p1 = Process(target=long_time_task, args=(1,))
#     p2 = Process(target=long_time_task, args=(2,))
#     print('等待所有子进程完成。')
#     p1.start()
#     p2.start()
#     p1.join()
#     p2.join()
#     end = time.time()
#     print("总共用时{}秒".format((end - start)))


# from multiprocessing import Pool, cpu_count
# import os
# import time


# def long_time_task(i):
#     print('子进程: {} - 任务{}'.format(os.getpid(), i))
#     time.sleep(2)
#     print("结果: {}".format(8 ** 20))


# if __name__=='__main__':
#     print("CPU内核数:{}".format(cpu_count()))
#     print('当前母进程: {}'.format(os.getpid()))
#     start = time.time()
#     p = Pool(4)
#     for i in range(5):
#         p.apply_async(long_time_task, args=(i,))
#     print('等待所有子进程完成。')
#     p.close()
#     p.join()
#     end = time.time()
#     print("总共用时{}秒".format((end - start)))


# from multiprocessing import Process, Queue
# import os, time, random

# # 写数据进程执行的代码:
# def write(q):
#     print('Process to write: {}'.format(os.getpid()))
#     for value in ['A', 'B', 'C']:
#         print('Put %s to queue...' % value)
#         q.put(value)
#         time.sleep(random.random())

# # 读数据进程执行的代码:
# def read(q):
#     print('Process to read:{}'.format(os.getpid()))
#     while True:
#         value = q.get(True)
#         print('Get %s from queue.' % value)

# if __name__=='__main__':
#     # 父进程创建Queue，并传给各个子进程：
#     q = Queue()
#     pw = Process(target=write, args=(q,))
#     pr = Process(target=read, args=(q,))
#     # 启动子进程pw，写入:
#     pw.start()
#     # 启动子进程pr，读取:
#     pr.start()
#     # 等待pw结束:
#     pw.join()
#     # pr进程里是死循环，无法等待其结束，只能强行终止:
#     pr.terminate()


# import threading
# import time

# def long_time_task(i):
#     print('当前子线程: {} 任务{}'.format(threading.current_thread().name, i))
#     time.sleep(2)
#     print("结果: {}".format(8 ** 20))

# if __name__=='__main__':
#     start = time.time()
#     print('这是主线程：{}'.format(threading.current_thread().name))
#     thread_list = []
#     for i in range(1, 3):
#         t = threading.Thread(target=long_time_task, args=(i, ))
#         thread_list.append(t)

#     for t in thread_list:
#         t.start()

#     for t in thread_list:
#         t.join()

#     end = time.time()
#     print("总共用时{}秒".format((end - start)))

# --------------------------------------------------------------

# import threading
# import time

# def long_time_task():
#     print('当子线程: {}'.format(threading.current_thread().name))
#     time.sleep(2)
#     print("结果: {}".format(8 ** 20))

# if __name__=='__main__':
#     start = time.time()
#     print('这是主线程：{}'.format(threading.current_thread().name))
#     for i in range(5):
#         t = threading.Thread(target=long_time_task, args=())

#         # 如果我们希望一个主线程结束时不再执行子线程，我们应该怎么办呢? 我们可以使用t.setDaemon(True)
#         t.setDaemon(True)
#         t.start()

#     end = time.time()
#     print("总共用时{}秒".format((end - start)))

# --------------------------------------------------------------

#-*- encoding:utf-8 -*-
import threading
import time


def long_time_task(i):
    time.sleep(2)
    return 8**20


class MyThread(threading.Thread):
    def __init__(self, func, args , name='', ):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.name = name
        self.result = None

    def run(self):
        print('开始子进程{}'.format(self.name))
        self.result = self.func(self.args[0],)
        print("结果: {}".format(self.result))
        print('结束子进程{}'.format(self.name))


if __name__=='__main__':
    start = time.time()
    threads = []
    for i in range(1, 3):
        t = MyThread(long_time_task, (i,), str(i))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    end = time.time()
    print("总共用时{}秒".format((end - start)))