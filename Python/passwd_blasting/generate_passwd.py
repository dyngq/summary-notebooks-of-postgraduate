# dict
import numpy as np

pw_01 = []
pw_02 = []
pw_03 = []
pw_04 = []

all_list = []

for i in pw_01:
    for j in pw_02:
        for k in pw_03:
            all_list.append(i+j+k)
            all_list.append(i+k+j)
            all_list.append(k+i+j)
            all_list.append(k+j+i)
            all_list.append(j+i+k)
            all_list.append(j+k+i)
            # print(i+j+k)
for i in pw_02:
    for j in pw_03:
        for k in pw_04:
            all_list.append(i+j+k)
            all_list.append(i+k+j)
            all_list.append(k+i+j)
            all_list.append(k+j+i)
            all_list.append(j+i+k)
            all_list.append(j+k+i)

# print(all_list)
with open('student.txt',mode='w') as f:
    for i in all_list:
        f.write(i+'\n')
