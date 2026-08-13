import json
import csv

file = "llama2-7b-adv391"  # specify the file name without .json extension
f = open('{}.json'.format(file), 'r')

data = json.load(f)

times = 0
cnt = 0
outputs = []
for key in data:
    times += data[key]['total_time']
    cnt += 1

tcps = times / cnt
print("time: ",tcps)


