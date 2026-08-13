import json
import csv

prompts = []
with open('icl_4_jaccard_refine.json', 'r', encoding='utf-8') as f:
    datas = json.load(f)
    for key, value in datas.items():
        prompts.append(value)
# print("prompt 1: ", prompts[0])
print("len: ", len(prompts))

with open("reda_prompt_120.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["origin_prompt", "final_prompt"])
    for prompt in prompts:
        writer.writerow(["", prompt])