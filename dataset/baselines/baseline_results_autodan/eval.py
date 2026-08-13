import json

file = "llama2-7b-adv391.json"
f = open(file, 'r')
data = json.load(f)
total = len(data)
success = 0
for key in data:
    if data[key]['is_success'] == True:
        success += 1
print(f"Success rate: {success}/{total} = {success/total:.4f}")