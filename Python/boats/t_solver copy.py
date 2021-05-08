import numpy as np
import math

a = np.random.randint(-30,30,[2])
b = np.random.randint(-30,30,[2])

print(a,b)

arc = 0.2
v = 5

def get_t(a,b,arc1,v):
    dis = math.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)
    b1 = dis*arc
    b2 = math.sqrt(dis**2-b1**2) 
    t1 = (b2*v-math.sqrt(b2**2*v**2-(v**2-130**2)*(b2**2-b1**2)))/(v**2-130**2)

    return t1

t = get_t(a,b,arc,v)

print(t)