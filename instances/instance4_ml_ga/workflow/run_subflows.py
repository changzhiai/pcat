import json, toml, sys
from pathlib import Path
from myqueue.workflow import run
from contextlib import contextmanager
import os 
import argparse
import time
import pandas as pd
import logging
import types
from datetime import datetime

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Workflow", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )
    return parser.parse_args(arg_list)

class Params:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Params(**value)
            else:
                self.__dict__[key] = value

def log_newline(self, lines=1):
    self.h.setFormatter(self.blank_formatter)
    for i in range(lines):
        self.info('')
    self.h.setFormatter(self.formatter)

def create_file_logger(name='watch.log', remove_basic_log=False):
    if remove_basic_log and os.path.exists(name):
        os.remove(name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    blank_formatter = logging.Formatter('')
    fh = logging.FileHandler(name) # file handler 
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.h = fh
    logger.formatter = formatter
    logger.blank_formatter = blank_formatter
    logger.newline = types.MethodType(log_newline, logger)
    return logger

@contextmanager
def cd(newdir, new=True):
    prevdir = os.getcwd()
    if new == True:
        try:
            os.makedirs(newdir)
        except OSError:
            pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

with open('config.toml') as f:
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
    need_lock = True
    dry_run = False
    name = name
    folder = folder
    with Queue(config, need_lock=need_lock, dry_run=dry_run) as queue:
        rows = queue.sql(
                'SELECT id, state FROM tasks WHERE name = ? AND folder = ?',
                [name, folder]
                )
        id, state = max(rows, default=(-1, 'u'))
        # print(id, state)
    return state

def get_name_and_folder(filename):
    df = pd.read_csv(filename)
    name = df['names'][0]
    folder = df['folders'][0]
    current_dir = args['global']['current_dir']
    current_dir = os.path.abspath(current_dir)
    folder = folder.replace(current_dir, '.')
    folder = folder + '/'
    # print(name, folder)
    return name, folder
 
def check_states(filename):
    df = pd.read_csv(filename)
    states = []
    names = []
    folders = []
    for index, row in df.iterrows():
        name = row['names']
        folder = row['folders']
        current_dir = args['global']['current_dir']
        current_dir = os.path.abspath(current_dir)
        folder = folder.replace(current_dir, '.')
        folder = folder + '/'
        state = check_state(name, folder)
        states.append(state=='d')
        names.append(name)
        folders.append(folder)
        # print(name, folder)
    return all(states), names, folders
 
def wait_until_done(name, folder, checktime=10, maxtime=7):
    """Wait until the tasks are done
    checktime unit is minute
    maxtime unit is day"""
    checktime = checktime # minutes
    maxtime = maxtime # days
    checktime *= 60
    maxtime *= 3600*24
    logger.info('Checking starting...')
    start = datetime.now()
    elapsed = datetime.now() - start
    elapsedseconds = elapsed.total_seconds()
    while elapsedseconds < maxtime:
        done = check_state(name, folder)
        # print(done)
        if done == 'd':
            break
        elapsed = datetime.now() - start
        elapsedseconds = elapsed.total_seconds()
        logger.info('Checking. Elapsed time: {:.2f} mins'
            .format(elapsedseconds / 60. ))
        time.sleep(checktime)
    return True

def wait_until_dones(names, folders):
    dones = []
    for name, folder in zip(names, folders):
        done = wait_until_done(name, folder)
        dones.append(done)
    return all(dones)

def run_workflows():
    deps = []
    print('========total start========\n')
    for iteration in range(33, 35):
    # for iteration in range(args["global"]["start_iteration"], 11):
        args['global']['current_iteration'] = iteration
        with open('config_temp.toml', 'w') as f:
            toml.dump(args, f)
        
        logger.info(f'...Starting iteration {iteration}...')
        """train-predict-ga"""
        logger.info('Starting subflow1...')
        os.system('mq workflow subflow1.py')
        time.sleep(5)
        name, folder = get_name_and_folder(filename='./iter_0/temp/task_gaing.csv')
        done = check_state(name, folder)
        print(f'subflow1 state: {done}')
        if done != 'd':
            wait_until_done(name, folder)
        logger.info('Finishing subflow1...')
        
        """pick"""
        logger.info('Starting subflow2...')
        os.system('mq workflow subflow2.py')
        time.sleep(5)
        name, folder = get_name_and_folder(filename='./iter_0/temp/task_picking.csv')
        done = check_state(name, folder)
        print(f'subflow2 state: {done}')
        if done != 'd':
            wait_until_done(name, folder)
        logger.info('Finishing subflow2...')
        
        """dft"""
        logger.info('Starting subflow3...')
        os.system('mq workflow subflow3.py')
        time.sleep(5)
        done, names, folders = check_states(filename='./iter_0/temp/task_dfting.csv')
        print(f'subflow3 state: {done}')
        if not done:
            wait_until_dones(names, folders)
        logger.info('Finishing subflow3...')
        
        """collect"""
        logger.info('Starting subflow4...')
        os.system('mq workflow subflow4.py')
        time.sleep(5)
        name, folder = get_name_and_folder(filename='./iter_0/temp/task_collecting.csv')
        done = check_state(name, folder)
        print(f'subflow4 state: {done}')
        if done != 'd':
            wait_until_done(name, folder)
        logger.info('Finishing subflow4...')

        print(f'iteration {iteration} done\n') 
        logger.info(f'...Finishing iteration {iteration}...')
        # break
    print(f'========totol done!========')

if __name__ == '__main__':
    logger = create_file_logger(name='watch.log')
    run_workflows()
