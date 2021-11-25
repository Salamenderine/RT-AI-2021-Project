import os

with open('test_command.txt') as f:
    for line in f.readlines():
        os.system(line.strip())