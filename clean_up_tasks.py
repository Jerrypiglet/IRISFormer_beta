from pathlib import Path
from tqdm import tqdm
from icecream import ic
import shutil

list_path = 'clean_up_tasks.txt'

folders = ['Checkpoint', 'logs', 'Summary_vis']

with open(list_path) as f:
    mylist = f.read().splitlines() 

mylist = [x.strip() for x in mylist]

for folder in folders:
    log_paths = Path(folder).iterdir()
    # print([x.name for x in log_paths])
    for log_path in tqdm(log_paths):
        task_name = log_path.name
        for task_datetime in mylist:
            if task_name.startswith(task_datetime):
                # Path(log_path).unlink()
                shutil.rmtree(log_path)
                print('Removed '+str(log_path))
