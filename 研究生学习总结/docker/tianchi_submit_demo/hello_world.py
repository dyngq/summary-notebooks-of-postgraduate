# import pandas as pd
import numpy as np
import json

# df = pd.read_csv("/tcdata/num_list.csv")
# df = pd.read_csv("./docker/tianchi_submit_demo/data/tcdata/num_list.csv")
# print(df)

numbers = np.loadtxt(open("./tcdata/num_list.csv","rb"),delimiter=",",skiprows=0,dtype='int')
# numbers = np.loadtxt(open("./docker/tianchi_submit_demo/data/tcdata/num_list.csv","rb"),delimiter=",",skiprows=0,dtype='int')

# numbers = np.random.randint(1,30,size=50,dtype='int32')
# print(numbers)
# np.savetxt('./docker/tianchi_submit_demo/data/tcdata/num_list.csv', numbers,delimiter = ',')

# print("hello_world")

# print(numbers,type(numbers.tolist()))

r_sum = np.sum(numbers)

top10 = numbers[np.argpartition(numbers,-10)[-10:]]
top10 = np.sort(top10).tolist()
top10.reverse()
# print(top10, type(top10))

result = {
    "Q1": "Hello world",
    "Q2": r_sum.tolist(),
    # C004 注意：TOP10 若包含重复值
    "Q3": top10
}
with open("result.json", "w") as f:
    json.dump(result, f) 