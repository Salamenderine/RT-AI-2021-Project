import os

files= os.listdir('../mnist_nets')
for f in files:
    tests = os.listdir('../test_cases/' + f[:-3])
    for test in tests:
        print(f[:-3], test, sep=' ')
        os.system('python verifier.py --net ' + f[:-3] + ' --spec ' + os.path.join('..', 'test_cases', f[:-3], test))

# with open('test_command.txt') as f:
#     for line in f.readlines():
#         # print(line.split(''))
#         os.system(line.strip())