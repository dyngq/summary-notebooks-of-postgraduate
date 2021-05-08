import torch
# x = torch.empty(5, 3)
# print(x)

# x = torch.rand(5, 3)
# print(x)

# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)

x = torch.tensor([5.5, 3])
print(x)

# 返回的tensor默认具有相同的torch.dtype和torch.device
x = x.new_ones(5, 3, dtype=torch.float64)  
print(x)

# 指定新的数据类型 (改变数据类型，size相同，数据变为随机值)
x = torch.randn_like(x, dtype=torch.float) 
print(x) 

print(x.size())
print(x.shape)

y = torch.rand(5, 3)
print(y)

print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

import numpy as np

def find_sub_max(arr, n):
    top_n = []
    for i in range(n-1):
        temp = np.argmax(arr)
        top_n.append(temp)
        arr_ = arr
        arr_[np.argmax(arr_)] = np.min(arr)
        arr = arr_
    return top_n
    
def wh_in_list(arr,n):
    return 0


if __name__ == '__main__':
    arr = [2, 3, 1, 7, 6, 5]
    # arr_bk = arr.copy()
    # res_top_n = find_sub_max(arr, 3)
    # print(res_top_n)
    # for i in res_top_n:
    #     print(arr_bk[i])
    a = 0
    x = 1
    while x in arr:
        a = a + 1
        x = x + 1
        print(a, x)
