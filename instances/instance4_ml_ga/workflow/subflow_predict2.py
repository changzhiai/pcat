import json, toml, sys
from pathlib import Path
from myqueue.workflow import run
from typing import List, Dict
from ase.io import Trajectory, read, write
from ase.constraints import FixAtoms
import numpy as np
import copy
import shutil
from contextlib import contextmanager
import copy
import os 

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

def train(folder, deps, extra_args: List[str] = [], iteration: int=0):
    """Train ensemble"""
    tasks = []
    current_dir = args['global']['current_dir']
    system = args['global']['system']
    start_iteration = args["global"]['start_iteration']
    current_dir = os.path.abspath(current_dir)
    collect_folder = args['collect']['folder']
    node_sizes = args['train']['node_sizes']
    node_info = args['train']['resource']
    if isinstance(node_sizes, list):
        if args['global']['verbose']:
            print('node_sizes:', node_sizes)
    else:
        raise ValueError('Only support ensemble calculator')
    for node_size in node_sizes:
        args_train = copy.deepcopy(args['train'])
        del args_train['node_sizes']
        last_iter_dir = f'{current_dir}/iter_{iteration-1}'
        run_dir = f'{current_dir}/iter_{iteration}/{folder}/{node_size}_nodes'
        with cd(run_dir):
            args_train['node_size'] = node_size
            # load model
            if iteration > start_iteration:
                load_model = f'{last_iter_dir}/{folder}/{node_size}_nodes/{args_train["output_dir"]}/best_model.pth'
                dataset = f'{last_iter_dir}/{collect_folder}/dft_{system}_adss_r{start_iteration-1}_to_r{iteration-1}_spc_undistor.traj'
                if os.path.isfile(load_model) and os.path.isfile(dataset):
                    args_train['load_model'] = load_model
                    args_train['dataset'] = dataset
                    if args['global']['verbose']:
                        print(f'load models: {args_train["dataset"]}')
                else:
                    raise ValueError(f'Module path does not exist. Error happens in {run_dir}. \n\
                            load model is {load_model} and dataset is {dataset}.')
            else:
                args_train['dataset'] = args_train['init_dataset']
                del args_train['init_dataset']
            
            args_train['system'] = system
            args_train['printlog_dir'] = f'{run_dir}/{args_train["output_dir"]}/{args_train["printlog"]}'
            args_train['run_dir'] = run_dir
            with open('arguments.toml', 'w') as f:
                toml.dump(args_train, f)

            arguments = ['--cfg', 'arguments.toml']                
            arguments += extra_args
            tasks.append(run(
                script=f'{current_dir}/train.py', 
                nodename='sm3090' if not node_info.get('nodename') else node_info['nodename'],
                cores=8 if not node_info.get('cores') else node_info['cores'], 
                tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
                args=arguments,
                folder=run_dir,
                name='train',
                deps=deps,
            ))
    return tasks

def predict(folder, deps, extra_args: List[str] = [], iteration: int=0):
    """Predict average ensemble energy via relaxing structures using ensemble method"""
    tasks = []
    folder2 = 'predict3'
    current_dir = args['global']['current_dir']
    start_iteration = args["global"]['start_iteration']
    system = args['global']['system']
    collect_folder = args['collect']['folder']
    current_dir = os.path.abspath(current_dir)
    node_info = args['predict']['resource']
    node_sizes = args['train']['node_sizes']
    if isinstance(node_sizes, list):
        if args['global']['verbose']:
            print('node_sizes:', node_sizes)
    else:
        raise ValueError('Only support ensemble calculator')
    load_models = []
    last_iter_dir = f'{current_dir}/iter_{iteration-1}'
    run_dir = f'{current_dir}/iter_{iteration}/{folder2}'
    with cd(run_dir):
        args_predict = copy.deepcopy(args['predict'])
        for node_size in node_sizes:
            # load model
            load_model = f'{current_dir}/iter_{iteration}/{args["train"]["folder"]}/{node_size}_nodes/{args["train"]["output_dir"]}/best_model.pth'
            # if not os.path.isfile(load_model):
                # raise ValueError(f'Module path does not exist. Error happens in {run_dir}. \nload model is {load_model}.')
            load_models.append(load_model)
        
        if iteration > start_iteration:
            all_init_dft_dir = f'{last_iter_dir}/{collect_folder}/dft_{system}_adss_r{start_iteration-1}_to_r{iteration-1}_init_tot.traj'
            all_final_dft_dir = f'{last_iter_dir}/{collect_folder}/dft_{system}_adss_r{start_iteration-1}_to_r{iteration-1}_final_tot.traj'
            if os.path.isfile(all_init_dft_dir) and os.path.isfile(all_final_dft_dir):
                args_predict['all_init_dft_dir'] = all_init_dft_dir
                args_predict['all_final_dft_dir'] = all_final_dft_dir
        else:
            args_predict['all_init_dft_dir'] = args_predict['init_traj'] 
            args_predict['all_final_dft_dir'] = args_predict['final_traj']

        args_predict['system'] = system
        args_predict['load_models'] = load_models
        args_predict['random_seed'] = args['train']['random_seed']
        args_predict['run_dir'] = run_dir
        with open('arguments.toml', 'w') as f:
            toml.dump(args_predict, f)

        arguments = ['--cfg', 'arguments.toml']                
        arguments += extra_args
        tasks.append(run(
            script=f'{current_dir}/predict2.py', 
            nodename='sm3090' if not node_info.get('nodename') else node_info['nodename'],
            cores=8 if not node_info.get('cores') else node_info['cores'], 
            tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
            args=arguments,
            folder=run_dir,
            name='predict',
            deps=deps,
        ))
    return tasks

def ga(folder, deps, extra_args: List[str] = [], iteration: int=0):
    """Genetic algorithm using ensemble calculator"""
    tasks = []
    current_dir = args['global']['current_dir']
    system = args['global']['system']
    start_iteration = args["global"]['start_iteration']
    current_dir = os.path.abspath(current_dir)
    node_sizes = args['train']['node_sizes']
    if isinstance(node_sizes, list):
        if args['global']['verbose']:
            print('node_sizes:', node_sizes)
    else:
        raise ValueError('Only support ensemble calculator')
    device = args['ga']['device']
    node_info = args['ga'][device]
    # parse parameters
    load_models = []
    last_iter_dir = f'{current_dir}/iter_{iteration-1}'
    run_dir = f'{current_dir}/iter_{iteration}/{folder}'
    with cd(run_dir):
        for node_size in node_sizes:
            # load model
            load_model = f'{current_dir}/iter_{iteration}/{args["train"]["folder"]}/{node_size}_nodes/{args["train"]["output_dir"]}/best_model.pth'
            if os.path.isfile(load_model):
                if args['global']['verbose']:
                    print(f'loading model: {load_model}')
            # else:
                # raise ValueError('Module path does not exist')
            load_models.append(load_model)
       
        args_ga = copy.deepcopy(args['ga'])
        args_ga['system'] = system
        args_ga['iteration'] = iteration
        args_ga['start_iteration'] = start_iteration
        args_ga['load_models'] = load_models
        args_ga['random_seed'] = args['train']['random_seed']
        args_ga['run_dir'] = run_dir
        args_ga['last_gen_traj'] = f'{last_iter_dir}/{args["pick"]["folder"]}/ga_{system}_adss_r{iteration-1}_candidates.traj'
        args_ga['last_db_dir'] =  f'{last_iter_dir}/{folder}/{args_ga["db_name"]}'
        if device == 'cpu':
            del args_ga['cuda']
        elif device == 'cuda':
            del args_ga['cpu']
        with open('arguments.toml', 'w') as f:
            toml.dump(args_ga, f)

        arguments = ['--cfg', 'arguments.toml']                
        arguments += extra_args
        tasks.append(run(
            script=f'{current_dir}/ga.py', 
            nodename='sm3090' if not node_info.get('nodename') else node_info['nodename'],
            cores=8 if not node_info.get('cores') else node_info['cores'], 
            tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
            args=arguments,
            folder=run_dir,
            name='ga',
            deps=deps,
        ))
    return tasks

def pick(folder, deps, extra_args: List[str] = [], iteration: int=0):
    """Select images from the result of GA and generate candidates"""
    tasks = []
    current_dir = args['global']['current_dir']
    system = args['global']['system']
    collect_folder = args['collect']['folder']
    start_iteration = args["global"]['start_iteration']
    current_dir = os.path.abspath(current_dir)
    node_info = args['pick']["resource"]
    # parse parameters
    load_models = []
    last_iter_dir = f'{current_dir}/iter_{iteration-1}'
    run_dir = f'{current_dir}/iter_{iteration}/{folder}'
    with cd(f'{current_dir}/iter_{iteration}/{folder}'):
        ga_folder = f'{current_dir}/iter_{iteration}/{args["ga"]["folder"]}' 
        args_pick = copy.deepcopy(args['pick'])
        args_pick['system'] = system
        args_pick['ga_db'] = f'{ga_folder}/{args["ga"]["db_name"]}'
        args_pick['fittest_images'] = f'{ga_folder}/{args["ga"]["fittest_images"]}'
        args_pick['last_gen_images'] = f'{ga_folder}/{args["ga"]["last_gen_images"]}'
        args_pick['pop_size'] = args["ga"]["pop_size"]
        args_pick['start_iteration'] = start_iteration
        args_pick['iteration'] = iteration
        args_pick['init_final_dft_images'] = args['predict']['final_traj']
        args_pick['all_old_dft_final_images'] = f'{last_iter_dir}/{collect_folder}/dft_{system}_adss_r{start_iteration-1}_to_r{iteration-1}_final_tot.traj'
        args_pick['pick_candidates_traj'] = f'ga_{system}_adss_r{iteration}_candidates.traj'
        args_pick['run_dir'] = run_dir
        with open('arguments.toml', 'w') as f:
            toml.dump(args_pick, f)

        arguments = ['--cfg', 'arguments.toml']                
        arguments += extra_args
        
        tasks.append(run(
            script=f'{current_dir}/pick.py', 
            nodename='sm3090' if not node_info.get('nodename') else node_info['nodename'],
            cores=8 if not node_info.get('cores') else node_info['cores'], 
            tmax='7d' if not node_info.get('tmax') else node_info['tmax'],
            args=arguments,
            folder=run_dir,
            name='pick',
            deps=deps,
        ))
    return tasks

def dft(folder, deps, extra_args: List[str] = [], iteration: int=0):
    """Run dft for candidates from pick"""
    tasks = []
    global args
    current_dir = args['global']['current_dir']
    system = args['global']['system']
    current_dir = os.path.abspath(current_dir)
    node_info = args['dft']['resource']
    # parse parameters
    load_models = []
    with cd(f'{current_dir}/iter_{iteration}/{folder}'):
        pick_folder = f'{current_dir}/iter_{iteration}/{args["pick"]["folder"]}' 
        args_dft = copy.deepcopy(args['dft'])
        args_dft['system'] = system
        args_dft['pick_candidates_traj'] = f'{pick_folder}/ga_{system}_adss_r{iteration}_candidates.traj'
        # print(f'deps: {deps}')
        if True:
            shutil.copy(args_dft['pick_candidates_traj'], './')
            images = read(args_dft['pick_candidates_traj'], ':')

            args_dft_tot = copy.deepcopy(args_dft)
            args_dft_tot['total_nums'] = len(images)
            with open('dft_total_info.toml', 'w') as f:
                toml.dump(args_dft_tot, f)

            for row_id, atoms in enumerate(images):
                if args_dft['only_start_one']:
                    if row_id != 0:
                        continue
                name = atoms.get_chemical_formula(mode='metal')
                job_id = str(row_id) + '_' + name
                run_dir = f'{current_dir}/iter_{iteration}/{folder}/{job_id}'
                with cd(job_id):
                    atoms_temp = atoms[[atom.index for atom in atoms if atom.tag >= 0]]
                    args_dft_temp = copy.deepcopy(args_dft)
                    args_dft_temp['job_id'] = job_id
                    try:
                        args_dft_temp['ads_indices'] = atoms.info['data']['ads_indices']
                        args_dft_temp['ads_symbols'] = atoms.info['data']['ads_symbols']
                    except:
                        args_dft_temp['ads_indices'] = atoms.info['ads_indices']
                        args_dft_temp['ads_symbols'] = atoms.info['ads_symbols']

                    args_dft_temp['slab'] = atoms_temp.get_chemical_formula(mode='metal') 
                    args_dft_temp['run_dir'] = run_dir
                    
                    with open('arguments.toml', 'w') as f:
                        toml.dump(args_dft_temp, f)
                    atoms.set_constraint(FixAtoms(mask=[atom.z<=args_dft_temp['fix_z_max'] for atom in atoms]))
                    write(args_dft['start_poscar'], atoms)
                    write(args_dft['start_traj'], atoms)

                    arguments = ['--cfg', 'arguments.toml']                
                    arguments += extra_args
                    tasks.append(run(
                        script=f'{current_dir}/dft.py', 
                        nodename='xeon24' if not node_info.get('nodename') else node_info['nodename'],
                        cores=24 if not node_info.get('cores') else node_info['cores'], 
                        tmax='2d' if not node_info.get('tmax') else node_info['tmax'],
                        args=arguments,
                        folder=run_dir,
                        name='dft',
                        deps=deps,
                    ))
                # if all_done(tasks):
                    # collecting = collect(args["collect"]["folder"], deps=tasks, iteration=iteration)
                    # print(f'-{args["collect"]["folder"]}: {iteration}')
        return tasks

def collect(folder, deps, extra_args: List[str] = [], iteration: int=0):
    """Run dft for candidates from pick"""
    # if all_done(deps):
    tasks = []
    global args
    current_dir = args['global']['current_dir']
    system = args['global']['system']
    start_iteration = args["global"]['start_iteration']
    current_dir = os.path.abspath(current_dir)
    node_info = args['collect']['resource']
    # parse parameters
    load_models = []
    last_iter_dir = f'{current_dir}/iter_{iteration-1}'
    run_dir = f'{current_dir}/iter_{iteration}/{folder}'
    with cd(run_dir):
        dft_dir = f'{current_dir}/iter_{iteration}/{args["dft"]["folder"]}'
        args_collect = copy.deepcopy(args['collect'])
        args_collect['system'] = system
        args_collect['start_iteration'] = start_iteration 
        args_collect['iteration'] = iteration
        args_collect['run_dir'] = run_dir
        args_collect['dft_dir'] = dft_dir
        args_collect['root_dir'] = current_dir
        args_collect['init_spc_dft_images'] = args['train']['init_dataset']
        args_collect['init_final_dft_images'] = args['predict']['final_traj']
        args_collect['init_init_dft_images'] = args['predict']['init_traj']
        args_collect['all_spc_dft_images'] = f'dft_{system}_adss_r{start_iteration-1}_to_r{iteration}_spc_undistor.traj' 
        args_collect['all_final_dft_images'] = f'dft_{system}_adss_r{start_iteration-1}_to_r{iteration}_final_tot.traj' 
        args_collect['all_init_dft_images'] = f'dft_{system}_adss_r{start_iteration-1}_to_r{iteration}_init_tot.traj' 
        args_collect['ga_candidates_traj'] = f'{dft_dir}/ga_{system}_adss_r{iteration}_candidates.traj'
        args_collect['dft_final_db'] = f'{run_dir}/dft_{system}_adss_r{iteration}.db'
        
        args_collect['csv'] = f'{run_dir}/dft_{system}_adss_r{iteration}_log.csv'        
        args_collect['images_spc'] = f'{run_dir}/dft_{system}_adss_r{iteration}_spc_tot.traj'
        args_collect['images_spc_undistor'] = f'{run_dir}/dft_{system}_adss_r{iteration}_spc_undistor.traj'
        args_collect['images_spc_distor'] = f'{run_dir}/dft_{system}_adss_r{iteration}_spc_distor.traj'
        args_collect['images_final'] = f'{run_dir}/dft_{system}_adss_r{iteration}_final_tot.traj'
        args_collect['images_final_undistor'] = f'{run_dir}/dft_{system}_adss_r{iteration}_final_undistor.traj'
        args_collect['images_final_distor'] = f'{run_dir}/dft_{system}_adss_r{iteration}_final_distor.traj'
        args_collect['images_init'] = f'{run_dir}/dft_{system}_adss_r{iteration}_init_tot.traj'
        args_collect['images_init_undistor'] = f'{run_dir}/dft_{system}_adss_r{iteration}_init_undistor.traj'
        args_collect['images_init_distor'] = f'{run_dir}/dft_{system}_adss_r{iteration}_init_distor.traj'
                
        with open('arguments.toml', 'w') as f:
            toml.dump(args_collect, f)
        arguments = ['--cfg', 'arguments.toml']                
        arguments += extra_args
        tasks.append(run(
            script=f'{current_dir}/collect.py', 
            nodename='xeon24' if not node_info.get('nodename') else node_info['nodename'],
            cores=24 if not node_info.get('cores') else node_info['cores'], 
            tmax='2d' if not node_info.get('tmax') else node_info['tmax'],
            args=arguments,
            folder=run_dir,
            name='collect',
            deps=deps,
                ))
    return tasks


def all_done(runs):
    return all([task.done for task in runs])

def workflow():
    deps = []
    print('start')
    for iteration in range(29, 32):
    # for iteration in range(args["global"]["start_iteration"], 18):
        """Train model and generate ensemble models"""
        # training = train(args["train"]["folder"], deps=deps, iteration=iteration)
        # print(f'-{args["train"]["folder"]}: {iteration}')
        
        """Predict energy and generate fitting plot"""
        predicting = predict(args["predict"]["folder"], deps=[], iteration=iteration)
        print(f'-{args["predict"]["folder"]}: {iteration}')
        
        """GA calculation and generate candidates"""
        # gaing = ga(args["ga"]["folder"], deps=predicting, iteration=iteration)
        # print(f'-{args["ga"]["folder"]}: {iteration}')
        
        # """Pick data from GA and generate candidates used to run dft"""
        # picking = pick(args["pick"]["folder"], deps=gaing, iteration=iteration)
        # print(f'-{args["pick"]["folder"]}: {iteration}')
        
        # """Run dft for picked trajectories and collect"""
        # dfting = dft(args["dft"]["folder"], deps=picking, iteration=iteration)
        # print(f'-{args["dft"]["folder"]}: {iteration}')
        
        # """Run dft for picked trajectories and collect"""
        # collecting = collect(args["collect"]["folder"], deps=dfting, iteration=iteration)
        # print(f'-{args["collect"]["folder"]}: {iteration}')

        print(f'Iteration {iteration} done!')

