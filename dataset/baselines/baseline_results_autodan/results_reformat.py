import json
import csv

file = "llama2-7b-adv391"  # specify the file name without .json extension
f = open('{}.json'.format(file), 'r')

data = json.load(f)

refine_prompts = []
outputs = []
for key in data:
    final_suffix = data[key]['final_suffix']
    final_suffix = final_suffix.replace('[REPLACE]', data[key]['goal'])
    refine_prompts.append(final_suffix)
    outputs.append(data[key]['final_respond'])

with open('{}.csv'.format(file), 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['final_prompt', 'output'])
    for prompt, output in zip(refine_prompts, outputs):
        writer.writerow([prompt, output])