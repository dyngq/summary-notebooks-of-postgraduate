import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

print(Model.__mro__)


# class A():
#     def fortest(self):
#         print('Call class A')
#         print('Leave class A')
# class B(A):
#     def fortest(self):
#         print('Call class B')
#         A.fortest(self)
#         print('Leave class B')
# sample=B()
# sample.fortest()
# print(B.__mro__)

# Call class B
# Call class A
# Leave class A
# Leave class B
# (<class '__main__.B'>, <class '__main__.A'>, <class 'object'>)

class A():
    def __init__(self):
        print('Call class A')
        super(A, self).__init__()
        print('Leave class A')
class B(A):
    def __init__(self):
        print('Call class B')
        super(B,self).__init__()
        print('Leave class B')
class C(A):
    def __init__(self):
        print('Call class C')
        super(C,self).__init__()
        print('Leave class C')
class D(A):
    def __init__(self):
        print('Call class D')
        super(D, self).__init__()
        print('Leave class D')
class E(B,C,D):
    def __init__(self):
        print('Call class E')
        super(E, self).__init__()
        print('Leave class E')
sample=E()
print(E.__mro__)