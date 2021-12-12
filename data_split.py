from random import shuffle

with open("CCPM-data/valid.jsonl", "r") as f:
    lines = list(f.readlines())

n = len(lines)
shuffle(lines)
val = lines[:n//2]
test = lines[n//2:]

with open("CCPM-data/split_valid.jsonl", "w") as f:
    f.writelines(val)

with open("CCPM-data/split_test.jsonl", "w") as f:
    f.writelines(test)