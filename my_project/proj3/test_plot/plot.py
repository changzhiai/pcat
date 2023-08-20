from ase.db import connect
from ase.io import read, write
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt

def generate_tasks(save_to_files=False):
    """Get reaction conditions"""
    T = np.array([298.15, 398.15])
    U = np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8])
    pH = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    P_H2 = np.array([101325.])
    P_CO2 = np.array([101325.])
    P_H2O = np.array([3534.])
    P_CO = np.array([5562.])
    kappa = np.array([0, 1, 2, 3, 4])
    cond_grid = np.array(np.meshgrid(T, U, pH, P_H2, P_CO2, P_H2O, P_CO, kappa)).T.reshape(-1, 8)
    df = pd.DataFrame(cond_grid, columns=['T', 'U', 'pH', 'P_H2', 'P_CO2', 'P_H2O', 'P_CO', 'kappa'])
    if save_to_files:
        df.to_pickle('em_tasks.pkl')
        df.to_csv('em_tasks.csv')
    return df

def generate_csv(fittest_images, save_to_csv=False):
    raw_niches = pd.read_pickle('../ga/em_tasks.pkl')
    dfs = []
    for i, atoms in enumerate(fittest_images):
        niches = atoms.info['data']['niches']
        scores = atoms.info['data']['raw_scores']
        dn = atoms.info['key_value_pairs']['dominating_niche']
        rs = atoms.info['key_value_pairs']['raw_score']
        print(atoms.info)
        # print(niches)
        # print(scores)
        # print(dn)
        # print(rs)
        # print(scores[1089])
        # for i, r in data.iterrows():
        # print(atoms.info['data']['raw_scores'])
        # print(scores[niches])
        data = copy.deepcopy(raw_niches)
        data['raw_scores'] = scores
        csv = str(i) + '_' + atoms.get_chemical_formula(mode='metal') + '.csv'
        # print(csv)
        if save_to_csv:
            data.to_csv(csv)
        dfs.append(data)
    return dfs

def basic_plot(x, y, xlabel, ft_sz=12):
    plt.figure()
    plt.plot(x, y, '-', c='blue', label=xlabel)
    plt.xlabel(xlabel, fontsize=ft_sz)
    plt.ylabel('Fitness function', fontsize=ft_sz)
    plt.legend()
    plt.show()

def plot_scores_vs_U(df):
    T = np.array([298.15, 398.15]).tolist()
    U = np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]).tolist()
    pH = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]).tolist()
    P_H2 = np.array([101325.]).tolist()
    P_CO2 = np.array([101325.]).tolist()
    P_H2O = np.array([3534.]).tolist()
    P_CO = np.array([5562.]).tolist()
    kappa = np.array([0, 1, 2, 3, 4]).tolist()
    df = df[(df['T']==T[0]) & (df['kappa']==kappa[1]) & (df['pH']==pH[0])]
    print(df)
    x_col = 'U'
    x = df[x_col]
    y = df['raw_scores']
    basic_plot(x, y, x_col, ft_sz=12)
    
def plot_scores_vs_pH(df):
    T = np.array([298.15, 398.15]).tolist()
    U = np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]).tolist()
    pH = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]).tolist()
    P_H2 = np.array([101325.]).tolist()
    P_CO2 = np.array([101325.]).tolist()
    P_H2O = np.array([3534.]).tolist()
    P_CO = np.array([5562.]).tolist()
    kappa = np.array([0, 1, 2, 3, 4]).tolist()
    df = df[(df['T']==T[0]) & (df['kappa']==kappa[1]) & (df['U']==pH[0])]
    print(df)
    x_col = 'pH'
    x = df[x_col]
    y = df['raw_scores']
    basic_plot(x, y, x_col, ft_sz=12)
    
        
if __name__ == '__main__':
    fittest_images = read('fittest_images.traj', ':')
    dfs = generate_csv(fittest_images, save_to_csv=False)
    df = dfs[2]
    plot_scores_vs_U(df)
    plot_scores_vs_pH(df)
    
    
