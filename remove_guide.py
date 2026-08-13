import csv

from path_utils import acl_data_path, side_data_path


origin_prompts = []
final_prompts = []
with open("/mnt/data/home/wangzhiyuan/SIDE/nlpcc_src/dataset/side/refine_jailbreakbench_prompt_100.csv", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        origin_prompts.append(row['origin_prompt'])
        final_prompt = row['final_prompt']
        final_prompt = final_prompt.replace('and begin your explanation with "Sure, here is the answer:', '')
        final_prompt = final_prompt.replace("The Process of a Criminal Committing the Crime:  <write detailed crime process here>.", '')
        final_prompt = final_prompt.replace("Laws and Regulations Violated by Offenders:  <write detailed laws and regulations violated by offenders here>.""", '')
        final_prompts.append(final_prompt)


with open("/mnt/data/home/wangzhiyuan/SIDE/nlpcc_src/dataset/side/remove_jailbreakbench_PGT.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['origin_prompt', 'final_prompt'])
    for origin_prompt, final_prompt in zip(origin_prompts, final_prompts):
        writer.writerow([origin_prompt, final_prompt])