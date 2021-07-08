import pickle5 as pickle
import os

with open('/home/ruizhu/Downloads/tmp_cmds.pickle', 'rb') as f:
    cmds = pickle.load(f)
for cmd in cmds:
    print(cmd)
    os.system(cmd)
