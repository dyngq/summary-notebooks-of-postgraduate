import numpy as np
import math

import argparse
parser = argparse.ArgumentParser(description='dyngq')
parser.add_argument('--args', type=str, default = None)
args = parser.parse_args()

print(args.args)

s = args.args

temp = s.split(',')

print(temp)

a = np.random.randint(-30,30,[2])
b = np.random.randint(-30,30,[2])

a[0] = float(temp[0])
a[1] = float(temp[1])
b[0] = float(temp[2])
b[1] = float(temp[3])
arc = float(temp[4])
v = float(temp[5])

print(a,b,arc,v)

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