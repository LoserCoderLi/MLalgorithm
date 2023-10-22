import numpy as np
import re
import pandas as pd

f = open("E:\scores.txt","r",encoding='utf-8')
a = f.read()
# f_1 = pd.read_csv("E:\scores.txt",header=None)
# print(f_1)
data = re.split('\ |\\n',a)

data = np.array(data)
data = data.reshape(-1,2)
print(data)
name = []
score = []

for count in range(len(data)):
    name.append(data[count][0])
    score.append(data[count][1])

dict = {}
for count in range(len(data)):
    if name[count] in dict:
        dict[name[count]] += [score[count]]
    else:
        dict[name[count]] = [score[count]]


dict['数学'] = sorted(dict['数学'],reverse=True)
dict['语文'] = sorted(dict['语文'],reverse=True)
dict['英语'] = sorted(dict['英语'],reverse=True)

dict_out = sorted(dict.items(),key=lambda x:x[1],reverse=True)
print(dict_out)


