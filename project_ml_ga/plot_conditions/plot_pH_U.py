from ase.db import connect
from ase.io import read, write
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import pcat.utils.constants as const

def generate_tasks(save_to_files=False):
    """Get reaction conditions"""
    T = np.array([298.15,])
    U = np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8])
    pH = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    P_H2 = np.array([101325.])
    P_CO2 = np.array([101325.])
    P_H2O = np.array([3534.])
    P_CO = np.array([5562.])
    kappa = np.array([0, 1, 2])
    cond_grid = np.array(np.meshgrid(T, U, pH, P_H2, P_CO2, P_H2O, P_CO, kappa)).T.reshape(-1, 8)
    df = pd.DataFrame(cond_grid, columns=['T', 'U', 'pH', 'P_H2', 'P_CO2', 'P_H2O', 'P_CO', 'kappa'])
    if save_to_files:
        df.to_pickle('./data/em_tasks.pkl')
        df.to_csv('./data/em_tasks.csv')
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
    print(kwargs)
    if 'plot_all' in kwargs and kwargs['plot_all'] == True:
        plt.plot(x, y, '-', c='grey', label=lable, linewidth=0.5)
    else:
        plt.plot(x, y, '-', c=c, label=lable, linewidth=1)
        # plt.legend()
    plt.xlabel(xlabel, fontsize=ft_sz)
    plt.ylabel('Surface free energy', fontsize=ft_sz)
    # plt.legend()
    plt.xlim([-0.8, 0.])
    # plt.ylim([-0.5, 0])
    plt.ylim([-0.35, -0.15])
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
    
# def plot_all_structs():
#     fittest_images = read('./data/fittest_images_critr5.traj', ':')
#     fittest_images2 = read('./data/fittest_images_1_6.traj', ':')
#     fittest_images3 = read('./data/last_1000.traj', ':')
#     fittest_images += fittest_images2
#     fittest_images += fittest_images3
#     dfs = generate_csv(fittest_images, raw_niches, save_to_csv=False)
#     plot_scores_vs_U_with_pHs(dfs, **{'plot_all': True})
        
if __name__ == '__main__':
    kappa = np.array([0, 1, 2])
    raw_niches = pd.read_pickle('./data/em_tasks.pkl')
    plt.figure(dpi=300)
    fittest_images = read('./data/fittest_images_critr5.traj', ':')
    fittest_images2 = read('./data/fittest_images_1_6.traj', ':')
    fittest_images += fittest_images2
    
    fittest_images = [atoms for i, atoms in enumerate(fittest_images) if i in [0, 2, 7, 8, 9, 10]]
    # raw_niches = pd.read_csv('em_tasks_all_conditions.csv')
    # fittest_images = read('fittest_images_all_contitions.traj', ':')
    # fittest_images = fittest_images[:3] + fittest_images[4:]
    # fittest_images = fittest_images[3]
    # plot_all_structs()
    dfs = generate_csv(fittest_images, raw_niches, save_to_csv=False)
    plot_scores_vs_U_with_pHs(dfs)
    # plot_scores_vs_pH_with_Us(dfs)
    plt.show()
    
    
