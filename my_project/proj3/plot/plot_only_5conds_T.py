from ase.db import connect
from ase.io import read, write
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import pcat.utils.constants as const
from ase.visualize import view

def generate_tasks(save_to_files=False):
    """Get reaction conditions"""
    d_mu_Pd = np.array([0, -0.25, -0.5, -0.75, -1.0])+(-2.24900)
    d_mu_Ti = np.array([0, -0.25, -0.5, -0.75, -1.0])+(-7.28453)
    d_mu_H = np.array([-3.61353])
    T = np.array([10, 25, 40, 65, 80])+273.15
    U = np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8])
    pH = np.array([0,])
    P_H2 = np.array([101325.])
    P_CO2 = np.array([101325.])
    P_H2O = np.array([3534.])
    # P_CO = np.logspace(-6, 0, 5)*101325.
    P_CO = np.array([0.101325, 10.1325, 1013.25, 5562, 101325])
    kappa = np.array([0, 1])
    keys = ['d_mu_Pd', 'd_mu_Ti', 'd_mu_H', 'T', 'U', 'pH', 'P_H2', 'P_CO2', 'P_H2O', 'P_CO', 'kappa']
    words = [d_mu_Pd, d_mu_Ti, d_mu_H, T, U, pH, P_H2, P_CO2, P_H2O, P_CO, kappa]
    df = {}
    for i in range(len(keys)):
        df[keys[i]] = words[i]  
    if save_to_files:
        cond_grid = np.array(np.meshgrid(d_mu_Pd, d_mu_Ti, d_mu_H, T, U, pH, P_H2, P_CO2, P_H2O, P_CO, kappa)).T.reshape(-1, 11)
        df = pd.DataFrame(cond_grid, columns=['d_mu_Pd', 'd_mu_Ti', 'd_mu_H', 'T', 'U', 'pH', 'P_H2', 'P_CO2', 'P_H2O', 'P_CO', 'kappa'])
        df.to_pickle('em_tasks.pkl')
        df.to_csv('em_tasks.csv')
    return df

def generate_csv(fittest_images, raw_niches, save_to_csv=False):
    dfs = []
    for i, atoms in enumerate(fittest_images):
        niches = atoms.info['data']['niches']
        scores = atoms.info['data']['raw_scores']
        dn = atoms.info['key_value_pairs']['dominating_niche']
        rs = atoms.info['key_value_pairs']['raw_score']
        # print(atoms.info['data']['raw_scores'])
        print(i, ':', scores[niches])
        print(set(raw_niches.iloc[niches]['U']))
        data = copy.deepcopy(raw_niches)
        data['raw_scores'] = scores # add a raw scores colomn
        csv = str(i) + '_' + atoms.get_chemical_formula(mode='metal') + '.csv'
        if save_to_csv:
            data.to_csv(csv)
        dfs.append(data)
    return dfs

def basic_plot(x, y, xlabel, c='C1', lable='image0', ft_sz=12, **kwargs):
    # plt.figure()
    # ax = plt.gca()
    # print(x, y)
    # x = x.values
    # y = y.values
    # ax.axline((x[0], y[0]), slope=(y[1]-y[0])/(x[1]-x[0]), c=c, label=lable)
    # print(kwargs)
    if 'plot_all' in kwargs and kwargs['plot_all'] == True:
        plt.plot(x, y, '-', c='grey', label=lable, linewidth=0.5)
    else:
        plt.plot(x, y, '-', c=c, label=lable, linewidth=1)
        # plt.legend()
    plt.xlabel(xlabel, fontsize=ft_sz)
    plt.ylabel('Surface free energy', fontsize=ft_sz)
    # plt.legend()
    # plt.xlim([-0.8, 0.])
    # plt.ylim([-0.35, -0.15])
    # plt.show()

def plot_scores_vs_U(df, i, target_pH, **kwargs):
    # df = df[(df['pH']==target_pH)]
    # df = df[(df['T']==T[0]) & (df['kappa']==kappa[1]) & (df['pH']==pH[0])]
    df = df[(df['kappa']==kappa[0]) & (df['pH']==target_pH)]
    # print(df)
    x_col = 'U'
    x = df[x_col]
    x += const.kB * df['T'] * target_pH * np.log(10)
    y = -df['raw_scores']
    basic_plot(x, y, x_col, c=f"C{i}", lable=f'image{i}', ft_sz=12, **kwargs)
    
def plot_multi_scores_vs_U(dfs, target_pH, **kwargs):
    # plt.figure(dpi=300)
    plt.title(f'pH value: {target_pH}',)
    for i in range(len(dfs)):
        df = dfs[i]
        plot_scores_vs_U(df, i, target_pH, **kwargs)
        # plot_scores_vs_pH(df)
    # plt.show()
    
def plot_scores_vs_U_with_pHs(dfs, **kwargs):
    """fix pH, plot scores vs. U"""
    pH = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]).tolist()
    only_pH_0 = True
    for target_pH in pH:
    # target_pH = pH[0]
        plot_multi_scores_vs_U(dfs, target_pH, **kwargs)
        if only_pH_0:
            break
    
def plot_scores_vs_pH(df, i, target_U, plot_all):
    # df = df[(df['T']==T[0]) & (df['kappa']==kappa[1]) & (df['U']==pH[0])]
    # print(df)
    df = df[(df['U']==target_U)]
    x_col = 'pH'
    x = df[x_col]
    y = -df['raw_scores']
    basic_plot(x, y, x_col, plot_all, c=f"C{i}", lable=f'image{i}', ft_sz=12)
    
def plot_multi_scores_vs_pH(dfs, target_U, plot_all=False):
    plt.figure(dpi=300)
    plt.title(f'U value: {target_U}',)
    for i in range(len(dfs)):
        df = dfs[i]
        plot_scores_vs_pH(df, i, target_U, plot_all)
        # plot_scores_vs_pH(df)
    plt.show()
    
def plot_scores_vs_pH_with_Us(dfs):
    """fix U, plot scores vs. pH"""
    U = np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]).tolist()
    for target_U in U:
    # target_pH = pH[0]
        plot_multi_scores_vs_pH(dfs, target_U)


def plot_scores_vs_chem(df, i, var, **kwargs):
    # df = df[(df['T']==T[0]) & (df['kappa']==kappa[1]) & (df['U']==pH[0])]
    # print(df)
    # df = df[(df['kappa']==kappa[0]) & (df['d_mu_Pd']==target_Pd_chem_pot)]

    df = df[(df['kappa']==kappa[0]) & (df['d_mu_Pd']==d_mu_Pd[0])  # change here
            & (df['d_mu_Ti']==d_mu_Ti[0]) & (df['T']==var)
            & (df['P_CO']==P_CO[3])]
    x_col = 'U'
    x = df[x_col]
    y = -df['raw_scores']
    basic_plot(x, y, x_col, c=f"C{i}", lable=f'image{i}', ft_sz=12, **kwargs)
    return x, y

def in_order(nums):
    ls = []
    for num in nums:
        if num not in ls:
            ls.append(num)
    return ls

def plot_multi_scores_vs_chem(dfs, var, ind, **kwargs):
    # fig = plt.figure(figsize=(8, 6), dpi=300)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    plt.title(f'Temperature {ind}: {var}',)
    results = pd.DataFrame(columns = U)
    for i in range(len(dfs)):
        df = dfs[i]
        x, y = plot_scores_vs_chem(df, i, var, **kwargs)
        assert (x.values == results.columns).all()
        results.loc[i] = y.values
    print(results)
    mini = results.min(axis=0)
    ids = results.idxmin(axis=0)
    id_set = in_order(ids)
    id_set = id_set[::-1] # reverse the order
    print(mini)
    print(ids)
    print(id_set)
    cand = [fittest_images[i] for i in id_set]
    # view(cand)
    plt.text(0.34, 0.03, id_set, horizontalalignment='left', verticalalignment='center',
             transform=ax.transAxes, fontsize=14, fontweight='bold')  
    write(f'T_{var}.traj', cand)
    # plt.legend()
    plt.show()
    return cand, id_set
        
def plot_scores_vs_chem_with_Us(dfs, **kwargs):
    """fix U, plot scores vs. pH"""
    vars = (T).tolist()
    cands, ids = [], []
    for ind, var in enumerate(vars):
        cand, id_set = plot_multi_scores_vs_chem(dfs, var, ind, **kwargs)
        cands.append(cand)
        ids.append(id_set)
    return cands, ids
    
def plot_all_structs():
    fittest_images = read('fittest_images_critr5.traj', ':')
    fittest_images2 = read('fittest_images_1_6.traj', ':')
    fittest_images3 = read('last_1000.traj', ':')
    fittest_images += fittest_images2
    fittest_images += fittest_images3
    dfs = generate_csv(fittest_images, raw_niches, save_to_csv=False)
    plot_scores_vs_U_with_pHs(dfs, **{'plot_all': True})
        
if __name__ == '__main__':
    df_raw = generate_tasks(save_to_files=False)
    kappa = df_raw['kappa']
    d_mu_Pd = df_raw['d_mu_Pd']
    U = df_raw['U']
    P_CO = df_raw['P_CO']
    T = df_raw['T']
    d_mu_Ti = df_raw['d_mu_Ti']
    raw_niches = pd.read_csv('./data/em_tasks_only_5conds.csv')
    plt.figure(dpi=300)
    fittest_images = read('./data/fittest_images_only_5conds.traj', ':')
    # fittest_images2 = read('fittest_images_1_6.traj', ':')
    # fittest_images += fittest_images2
    
    # fittest_images = [atoms for i, atoms in enumerate(fittest_images) if i in [0, 2, 7, 8, 9, 10]]
    # raw_niches = pd.read_csv('em_tasks_all_conditions.csv')
    # fittest_images = read('fittest_images_all_contitions.traj', ':')
    # fittest_images = fittest_images[:3] + fittest_images[4:]
    # fittest_images = fittest_images[3]
    # plot_all_structs()
    dfs = generate_csv(fittest_images, raw_niches, save_to_csv=False)
    # plot_scores_vs_U_with_pHs(dfs)
    # plot_scores_vs_pH_with_Us(dfs)
    cands, ids = plot_scores_vs_chem_with_Us(dfs, **{'plot_all': False})
    plt.show()
    
    
