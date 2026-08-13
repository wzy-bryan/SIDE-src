import csv

path1 = "harmful.csv"
path2 = "benign-behaviors.csv"

goal = []
with open(path1, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 去掉引号保存
        goal.append(row['Goal'])

benign = []
with open(path2, 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 去掉引号保存
        benign.append(row['Goal'])

with open('harmful_harmless_pair.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['harmful', 'harmless'])
    for i in range(min(len(goal), len(benign))):
        writer.writerow([goal[i], benign[i]])
        