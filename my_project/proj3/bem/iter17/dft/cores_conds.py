from ase.db import connect
from ase.io import read, write
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import pcat.utils.constants as const
from ase.visualize import view
import matplotlib.colors as mcolors
# print(mcolors.CSS4_COLORS)
import os

def basic_plot(x, y, xlabel, c='C1', lable='image0', ft_sz=12, **kwargs):
    # plt.figure()
    # ax = plt.gca()
    if 'gray_above' in kwargs and kwargs['gray_above'] == True:
        plt.plot(x, y, '-', c='grey', label=lable, linewidth=0.5)
    else:
        plt.plot(x, y, '-', c=c, label=lable, linewidth=1)
    # plt.xlabel(xlabel, fontsize=ft_sz)
    # plt.ylabel('Surface free energy', fontsize=ft_sz)
    # plt.legend()
    # plt.xlim([-0.8, 0.])
    # plt.ylim([-0.35, -0.15])
    # plt.show()

def plot_scores(df, i, d_mu_pd, d_mu_ti, pco, t, x_col = 'U', **kwargs):
    if 'df_raw' in kwargs:
        df_raw = kwargs['df_raw']
    kappa = df_raw['kappa'][0]

    df = df[(df['kappa']==kappa) & (df['d_mu_Pd']==d_mu_pd)  # change here
            & (df['d_mu_Ti']==d_mu_ti) & (df['T']==t) # x axis back and forward
            & (df['P_CO']==pco)]
    
    df = df.sort_values(x_col)
    x = df[x_col]
    y = -df['raw_scores']
    if 'iter' in kwargs:
        iter = kwargs['iter']
    # plt.title(f'Iter {iter}, Ti chem.: {round(d_mu_ti,3)}, Pd chem.: {d_mu_pd}, T: {t}K, Pco: {pco}Pa')
    basic_plot(x, y, x_col, c=f"C{i}", lable=f'image{i}', ft_sz=12, **kwargs)
    return x, y

def in_order(nums):
    ls = []
    for num in nums:
        if num not in ls:
            ls.append(num)
    return ls

def plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, ax=None, x_col = 'U', **kwargs):
    if 'iter' in kwargs:
        iter = kwargs['iter']
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    if 'df_raw' in kwargs:
        df_raw = kwargs['df_raw']
    U = df_raw['U']
    U = list(sorted(set(U)))
    results = pd.DataFrame(columns = U)
    for i in range(len(dfs)):
        df = dfs[i]
        x, y = plot_scores(df, i, d_mu_pd, d_mu_ti, pco, t, x_col = 'U', **kwargs)
        assert (x.values == results.columns.values).all()
        results.loc[i] = y.values
    print(results)
    minis = results.min(axis=0)
    ids = results.idxmin(axis=0)
    id_set = in_order(ids)
    # id_set = id_set[::-1] # reverse the order
    print('mini:', minis)
    print('ids:', ids)
    print('id_set:', id_set)
    if 'images' in kwargs:
        fittest_images = kwargs['images']
    cand = [fittest_images[i] for i in id_set]
    for i, id in enumerate(id_set):
        x = results.columns.values
        y = results.iloc[id].values
        x_col = 'U'
        basic_plot(x, y, x_col, c=f"C{i}", lable=f'image{i}', ft_sz=12,)
    return cand, id_set, minis, ids
        
def plot_surf_free_vs_U(dfs, **kwargs):
    """fix U, plot scores vs. U"""
    if 'iter' in kwargs:
        iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    T = kwargs['T']
    cands, ids = [], []
    # fig = plt.figure(dpi=300)
    for d_mu_pd in d_mu_Pd:
        for d_mu_ti in d_mu_Ti:
            for pco in P_CO:
                for t in T:
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                    cand, id_set, _, _ = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, ax=ax, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    plt.title(f'Iter {iter}, Ti chem.: {round(d_mu_ti,3)}, Pd chem.: {d_mu_pd}, T: {t}K, Pco: {pco}Pa')
                    plt.text(0.34, 0.03, id_set, horizontalalignment='left', verticalalignment='center',
                              transform=ax.transAxes, fontsize=14, fontweight='bold')
                    path = f'./figures/Pd_{d_mu_pd}_Ti_{d_mu_ti}'
                    if not(os.path.exists(path) and os.path.isdir(path)):
                        os.mkdir(path)
                    fig.savefig(f'{path}/iter_{iter}_Pd_{d_mu_pd}_Ti_{d_mu_ti}.png',dpi=300, bbox_inches='tight')
    plt.show()
    return cands, ids

def plot_surf_free_vs_U_matrix(dfs, **kwargs):
    """fix U, plot scores vs. U. only change chem. pot. of Pd and Ti"""
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    T = kwargs['T']
    cands, ids = [], []
    # fig = plt.figure(dpi=300)
    plt.figure(figsize=(12,10),dpi=300)
    i=0
    for d_mu_pd in d_mu_Pd:
        for d_mu_ti in d_mu_Ti:
            for pco in P_CO:
                for t in T:
                    ax = plt.subplot(5, 5, i+1)
                    cand, id_set, _, _ = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, ax, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 6
                    plt.text(0.05, 0.05, id_set, horizontalalignment='left', verticalalignment='center',
                             transform=ax.transAxes, fontsize=ft_sz)  
                    plt.text(0.05, 0.90,f'Ti: {round(d_mu_ti,3)}, Pd: {d_mu_pd}',
                             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,fontsize=ft_sz)
                    i += 1
                    ft_sz = 10
                    if i % 5 == 1:
                        plt.ylabel('Surface free energy', fontsize=ft_sz)
                    if i > 20:
                        plt.xlabel('U', fontsize=ft_sz)
    plt.show()
    return cands, ids

def plot_surf_free_vs_U_contour(dfs, **kwargs):
    """fix U, plot scores vs. U. only change chem. pot. of Pd and Ti"""
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    T = kwargs['T']
    U = kwargs['U']
    cands, ids = [], []
    # fig = plt.figure(dpi=300)
    plt.figure(figsize=(12,10),dpi=300)
    i=0
    for d_mu_pd in d_mu_Pd:
        for d_mu_ti in d_mu_Ti:
            for pco in P_CO:
                for t in T:
                    ax = plt.subplot(5, 5, i+1)
                    cand, id_set, minis, ids = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, ax, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 6
                    plt.text(0.05, 0.05, id_set, horizontalalignment='left', verticalalignment='center',
                             transform=ax.transAxes, fontsize=ft_sz)  
                    plt.text(0.05, 0.90,f'Ti: {round(d_mu_ti,3)}, Pd: {d_mu_pd}',
                             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,fontsize=ft_sz)
                    i += 1
                    ft_sz = 10
                    if i % 5 == 1:
                        plt.ylabel('Surface free energy', fontsize=ft_sz)
                    if i > 20:
                        plt.xlabel('U', fontsize=ft_sz)
    plt.show()
    return cands, ids
    
        
if __name__ == '__main__':
    ''

    
