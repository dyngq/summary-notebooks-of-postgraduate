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

