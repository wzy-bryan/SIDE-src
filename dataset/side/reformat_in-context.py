import csv


origin_prompts = []
final_prompts = []
outputs = []
with open('in-context_success.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        origin_prompts.append(row['origin_prompt'])
        final_prompts.append(row['final_prompt'])
        outputs.append((row['output'].replace('**', "")).replace('*   ', ''))

with open('in-context_success_refined.csv', 'w' ) as f:
    writer = csv.writer(f)
    writer.writerow(['origin_prompt', 'final_prompt', 'output'])
    for origin_prompt, final_prompt, output in zip(origin_prompts, final_prompts, outputs):
        writer.writerow([origin_prompt, final_prompt, output])
