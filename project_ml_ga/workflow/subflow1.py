import json, toml, sys
from pathlib import Path
import pandas as pd
from flow import train, predict, ga

with open('config_temp.toml') as f:
    args = toml.load(f)

def check_state(name, folder):
    """Check state for a task
    name = 'train'
    f = './iter_1/train/136_nodes/'
    """
    from myqueue.config import Configuration
    from pathlib import Path
    from myqueue.queue import Queue

    folders = [Path.cwd()]
    start = folders[0]
    config = Configuration.read(start)
    # print(config)
    need_lock = False
    dry_run = False
    name = 'train'
    f = './iter_1/train/136_nodes/'
    with Queue(config, need_lock=need_lock, dry_run=dry_run) as queue:
        rows = queue.sql(
                'SELECT id, state FROM tasks WHERE name = ? AND folder = ?',
                [name, f]
                )
        id, state = max(rows, default=(-1, 'u'))
        print(id, state)
    return state

def write_tasks(runs, filename):
    names, folders = [], []
    for run in runs:
        print(run.task)
        name = run.task.name.split('.')[0]
        folder = run.task.folder
        # state = check_state(name, folder)
        names.append(name)
        folders.append(folder)
    tuples = {
            'names': names,
            'folders': folders,
            }
    df = pd.DataFrame(tuples)
    df.to_csv(f'./iter_0/temp/{filename}.csv')
    return df

def workflow():
    deps = []
    # iteration = 10
    iteration = args['global']['current_iteration']
    print(f'======>{args["train"]["folder"]}: iteration {iteration}')
    training = train(args["train"]["folder"], deps=deps, iteration=iteration)
    write_tasks(training, 'task_training')
    
    print(f'======>{args["predict"]["folder"]}: iteration {iteration}')
    predicting = predict(args["predict"]["folder"], deps=training, iteration=iteration)
    write_tasks(predicting, 'task_predicting')
    
    print(f'======>{args["ga"]["folder"]}: iteration {iteration}')
    gaing = ga(args["ga"]["folder"], deps=predicting, iteration=iteration)
    write_tasks(gaing, 'task_gaing')
