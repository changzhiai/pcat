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
    if 'gray_above' in kwargs and kwargs['gray_above'] == None:
        return
    if 'gray_above' in kwargs and kwargs['gray_above'] == True:
        plt.plot(x, y, '-', c='grey', label=lable, linewidth=0.5)
    else:
        plt.plot(x, y, '-', c=c, label=lable, linewidth=1)

def plot_scores(df, i, d_mu_pd, d_mu_ti, pco, t, x_col='U', **kwargs):
    if 'df_raw' in kwargs:
        df_raw = kwargs['df_raw']
    kappa = df_raw['kappa'][0]
    df = df[(df['kappa']==kappa) & (df['d_mu_Pd']==d_mu_pd)  # change here
            & (df['d_mu_Ti']==d_mu_ti) & (df['T']==t) # x axis back and forward
            & (df['P_CO']==pco)]
    df = df.sort_values(x_col)
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

def get_colorslist(NUM_COLORS=1000):
    import random
    cm = plt.get_cmap(plt.cm.jet)
    cs = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    random.Random(666).shuffle(cs)
    return cs
            
def plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, x_col='U', **kwargs):
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
    print('results:',results)
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
        if 'gray_above' in kwargs and kwargs['gray_above'] == True:
            if True:
                basic_plot(x, y, x_col, c=f'C{i}', lable=f'image{i}', ft_sz=12)
            else:
                cs = get_colorslist(NUM_COLORS=1000)
                basic_plot(x, y, x_col, c=cs[id], lable=f'image{i}', ft_sz=12)
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
    path = './figures/pourbaix'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    # fig = plt.figure(dpi=300)
    for d_mu_ti in d_mu_Ti:
        for d_mu_pd in d_mu_Pd:
            for pco in P_CO:
                for t in T:
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                    cand, id_set, _, _ = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 12
                    plt.xlabel('Potential (V)', fontsize=ft_sz)
                    plt.ylabel('Surface free energy (eV)', fontsize=ft_sz)
                    plt.xlim([-0.8, 0.])
                    plt.ylim([-1., 0.251])
                    plt.title(f'Iter {iter}, $\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}, T: {t} K, Pco: {pco} Pa')
                    plt.text(0.34, 0.03, id_set, horizontalalignment='left', verticalalignment='center',
                              transform=ax.transAxes, fontsize=14, fontweight='bold')
                    path = f'./figures/pourbaix/Pd_{d_mu_pd}_Ti_{d_mu_ti}'
                    if not(os.path.exists(path) and os.path.isdir(path)):
                        os.mkdir(path)
                    fig.savefig(f'{path}/iter_{iter}_Pd_{d_mu_pd}_Ti_{d_mu_ti}.png',dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close(fig)
                    plt.clf()
    # plt.show()
    return cands, ids

def plot_surf_free_vs_U_matrix(dfs, **kwargs):
    """fix U, plot scores vs. U. only change chem. pot. of Pd and Ti"""
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    T = kwargs['T']
    iter = kwargs['iter']
    cands, ids = [], []
    fig = plt.figure(figsize=(12,10),dpi=300)
    i=0
    for d_mu_ti in d_mu_Ti:
        for d_mu_pd in d_mu_Pd:
            for pco in P_CO:
                for t in T:
                    ax = plt.subplot(5, 5, i+1)
                    cand, id_set, _, _ = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 6
                    plt.text(0.05, 0.05, id_set, horizontalalignment='left', verticalalignment='center',
                             transform=ax.transAxes, fontsize=ft_sz)  
                    plt.text(0.05, 0.90, f'$\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}',
                             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,fontsize=ft_sz)
                    plt.xlim([-0.8, 0.])
                    plt.ylim([-1., 0.25])
                    i += 1
                    ft_sz = 10
                    if i % 5 == 1:
                        plt.ylabel('Surface free energy', fontsize=ft_sz)
                        plt.yticks([-1.00, -0.75, -0.50, -0.25, 0.00])
                    else:
                        plt.yticks([])
                    if i > 20:
                        plt.xlabel('Potential (V)', fontsize=ft_sz)
                        plt.xticks([-0.6, -0.4, -0.2, 0.0])
                    else:
                        plt.xticks([])
    st = plt.suptitle(f'Iter: {iter}, T={T[0]} K, Pco={P_CO[0]} Pa', fontsize=ft_sz)
    fig.tight_layout(pad=0.2)
    st.set_y(1.01)
    plt.show()
    plt.close(fig)
    plt.clf()
    path = './figures/matrix_pourbaix'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    fig.savefig(f'{path}/iter_{iter}_matrix_pourbaix.png',dpi=300, bbox_inches='tight')
    return cands, ids

def plot_surf_free_vs_U_matrix_all(dfs, **kwargs):
    """Generate len(T)xlen(P_CO) 5x5 matrix plot"""
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    T = kwargs['T']
    iter = kwargs['iter']
    cands, ids = [], []
    path = './figures/matrix_all'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    path = f'./figures/matrix_all/iter_{iter}'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    for t in T:
        path = f'./figures/matrix_all/iter_{iter}/{t}'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        for pco in P_CO:
            fig = plt.figure(figsize=(12,10),dpi=300)
            i=0
            for d_mu_ti in d_mu_Ti:
                for d_mu_pd in d_mu_Pd:
                    ax = plt.subplot(5, 5, i+1)
                    cand, id_set, _, _ = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 6
                    plt.text(0.05, 0.05, id_set, horizontalalignment='left', verticalalignment='center',
                             transform=ax.transAxes, fontsize=ft_sz)  
                    plt.text(0.05, 0.90, f'$\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}',
                             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,fontsize=ft_sz)
                    plt.xlim([-0.8, 0.])
                    plt.ylim([-1., 0.25])
                    i += 1
                    ft_sz = 10
                    if i % 5 == 1:
                        plt.ylabel('Surface free energy', fontsize=ft_sz)
                        plt.yticks([-1.00, -0.75, -0.50, -0.25, 0.00])
                    else:
                        plt.yticks([])
                    if i > 20:
                        plt.xlabel('Potential (V)', fontsize=ft_sz)
                        plt.xticks([-0.6, -0.4, -0.2, 0.0])
                    else:
                        plt.xticks([])
            st = plt.suptitle(f'Iter: {iter}, T={t} K, Pco={pco} Pa', fontsize=ft_sz)
            fig.tight_layout(pad=0.2)
            st.set_y(1.01)
            plt.show()
            fig.savefig(f'{path}/iter_{iter}_matrix_all_T_{t}_Pco_{pco}.png',dpi=300, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
    return cands, ids

def plot_surf_free_vs_U_contour(dfs, **kwargs):
    """fix U, plot scores vs. U. only change chem. pot. of Pd and Ti
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    T = kwargs['T']
    df_raw = kwargs['df_raw']
    Us = sorted(set(df_raw['U']))
    minuss, idss = [], []
    for U in Us:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_pd in d_mu_Pd:
            for d_mu_ti in d_mu_Ti:
                for pco in P_CO:
                    for t in T:
                        cand, id_set, minis, ids = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                        xs.append(d_mu_pd)
                        ys.append(d_mu_ti)
                        zs.append(minis[U])
                        tags.append(ids[U])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(5,5), ys.reshape(5,5), zs.reshape(5,5))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(f'Iter. {iter}, U={U} V, T={T[0]} K, Pco={P_CO[0]} Pa', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        # plt.show()
        path = './figures/SFE'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_pot_{U}_SFE.png',dpi=300, bbox_inches='tight')
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        plt.xlim([-3.25, -2.25])
        plt.ylim([-8.28, -7.28])
        # cs = get_colorslist(NUM_COLORS=1000)
        # colors = []
        # for tag in tags:
        #     colors.append(cs[tag])
        # plt.scatter(xs, ys, c=colors, marker='s',s=2800, linewidths=4)
        # plt.xlim([-3.37, -2.13])
        # plt.ylim([-8.41, -7.16])
        # for i, txt in enumerate(tags):
        #     ax.annotate(txt, (xs[i], ys[i]), ha='center')
        ax.set_title(f'Iter. {iter}, U={U} V, T={T[0]} K, Pco={P_CO[0]} Pa', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        plt.show()
        path = './figures/cands'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_pot_{U}_cands.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss


def plot_surf_free_vs_Tichem_and_Pdchem(dfs, **kwargs):
    """fix U, plot scores vs. U. only change chem. pot. of Pd and Ti
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    T = kwargs['T']
    U = kwargs['U']
    images = kwargs['images']
    minuss, idss = [], []
    for u in U:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_pd in d_mu_Pd:
            for d_mu_ti in d_mu_Ti:
                for pco in P_CO:
                    for t in T:
                        cand, id_set, minis, ids = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                        xs.append(d_mu_pd)
                        ys.append(d_mu_ti)
                        zs.append(minis[u])
                        tags.append(ids[u])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(5,5), ys.reshape(5,5), zs.reshape(5,5))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(f'Iter. {iter}, U={u} V, T={T[0]} K, Pco={P_CO[0]} Pa', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        plt.show()
        path = './figures/Contour_vs_Tichem_and_Pdchem'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_pot_{u}_SFE.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            # label = images[label].info['formula']
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        plt.xlim([-3.25, -2.25])
        plt.ylim([-8.28, -7.28])
        # cs = get_colorslist(NUM_COLORS=1000)
        # colors = []
        # for tag in tags:
        #     colors.append(cs[tag])
        # plt.scatter(xs, ys, c=colors, marker='s',s=2800, linewidths=4)
        # plt.xlim([-3.37, -2.13])
        # plt.ylim([-8.41, -7.16])
        # for i, txt in enumerate(tags):
        #     ax.annotate(txt, (xs[i], ys[i]), ha='center')
        ax.set_title(f'Iter. {iter}, U={u} V, T={T[0]} K, Pco={P_CO[0]} Pa', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        plt.show()
        path = './figures/Heatmap_vs_Tichem_and_Pdchem'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_pot_{U}_cands.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss

# def basic_plot(x, y, xlabel, c='C1', lable='image0', ft_sz=12, **kwargs):
#     if 'gray_above' in kwargs and kwargs['gray_above'] == None:
#         return
#     if 'gray_above' in kwargs and kwargs['gray_above'] == True:
#         plt.plot(x, y, '-', c='grey', label=lable, linewidth=0.5)
#     else:
#         plt.plot(x, y, '-', c=c, label=lable, linewidth=1)

# T      
def plot_scores_T(df, i, d_mu_pd, d_mu_ti, pco, u, x_col='T', **kwargs):
    if 'df_raw' in kwargs:
        df_raw = kwargs['df_raw']
    kappa = df_raw['kappa'][0]
    df = df[(df['kappa']==kappa) & (df['d_mu_Pd']==d_mu_pd)  # change here
            & (df['d_mu_Ti']==d_mu_ti) & (df['U']==u) # x axis back and forward
            & (df['P_CO']==pco)]
    df = df.sort_values(x_col)
    x = df[x_col]
    y = -df['raw_scores']
    basic_plot(x, y, x_col, c=f"C{i}", lable=f'image{i}', ft_sz=12, **kwargs)
    return x, y

def plot_multi_scores_vs_T(dfs, d_mu_pd, d_mu_ti, pco, u, x_col='T', **kwargs):
    if 'df_raw' in kwargs:
        df_raw = kwargs['df_raw']
    T = df_raw['T']
    T = list(sorted(set(T)))
    results = pd.DataFrame(columns = T)
    for i in range(len(dfs)):
        df = dfs[i]
        x, y = plot_scores_T(df, i, d_mu_pd, d_mu_ti, pco, u, x_col = 'T', **kwargs)
        assert (x.values == results.columns.values).all()
        results.loc[i] = y.values
    print('results:',results)
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
        if 'gray_above' in kwargs and kwargs['gray_above'] == True:
            if True:
                basic_plot(x, y, x_col, c=f'C{i}', lable=f'image{i}', ft_sz=12)
            else:
                cs = get_colorslist(NUM_COLORS=1000)
                basic_plot(x, y, x_col, c=cs[id], lable=f'image{i}', ft_sz=12)
    return cand, id_set, minis, ids

def plot_surf_free_vs_T(dfs, **kwargs):
    """fix U, plot scores vs. U"""
    if 'iter' in kwargs:
        iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    U = kwargs['U']
    cands, ids = [], []
    path = './figures/pourbaix_T'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    # fig = plt.figure(dpi=300)
    for d_mu_ti in d_mu_Ti:
        for d_mu_pd in d_mu_Pd:
            for pco in P_CO:
                for u in U:
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                    cand, id_set, _, _ = plot_multi_scores_vs_T(dfs, d_mu_pd, d_mu_ti, pco, u, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 12
                    plt.xlabel('Temperature (K)', fontsize=ft_sz)
                    plt.ylabel('Surface free energy (eV)', fontsize=ft_sz)
                    plt.xlim([283.15, 353.15])
                    plt.ylim([-1., 0.251])
                    plt.title(f'Iter {iter}, $\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}, U: {u} V, Pco: {pco} Pa')
                    plt.text(0.34, 0.03, id_set, horizontalalignment='left', verticalalignment='center',
                              transform=ax.transAxes, fontsize=14, fontweight='bold')
                    path = f'./figures/pourbaix_T/Pd_{d_mu_pd}_Ti_{d_mu_ti}'
                    if not(os.path.exists(path) and os.path.isdir(path)):
                        os.mkdir(path)
                    fig.savefig(f'{path}/iter_{iter}_Pd_{d_mu_pd}_Ti_{d_mu_ti}.png',dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close(fig)
                    plt.clf()
    return cands, ids

def plot_surf_free_vs_T_matrix(dfs, **kwargs):
    """fix U, plot scores vs. U. only change chem. pot. of Pd and Ti"""
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    U = kwargs['U']
    iter = kwargs['iter']
    cands, ids = [], []
    fig = plt.figure(figsize=(12,10),dpi=300)
    i=0
    for d_mu_ti in d_mu_Ti:
        for d_mu_pd in d_mu_Pd:
            for pco in P_CO:
                for u in U:
                    ax = plt.subplot(5, 5, i+1)
                    cand, id_set, _, _ = plot_multi_scores_vs_T(dfs, d_mu_pd, d_mu_ti, pco, u, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 6
                    plt.text(0.05, 0.05, id_set, horizontalalignment='left', verticalalignment='center',
                             transform=ax.transAxes, fontsize=ft_sz)  
                    plt.text(0.05, 0.90, f'$\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}',
                             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,fontsize=ft_sz)
                    plt.xlim([283.15, 353.15])
                    plt.ylim([-1., 0.25])
                    i += 1
                    ft_sz = 10
                    if i % 5 == 1:
                        plt.ylabel('Surface free energy', fontsize=ft_sz)
                        plt.yticks([-1.00, -0.75, -0.50, -0.25, 0.00])
                    else:
                        plt.yticks([])
                    if i > 20:
                        plt.xlabel('Temperature (K)', fontsize=ft_sz)
                        # plt.xticks([-0.6, -0.4, -0.2, 0.0])
                    else:
                        plt.xticks([])
    st = plt.suptitle(f'Iter: {iter}, U={U[0]} V, Pco={P_CO[0]} Pa', fontsize=ft_sz)
    fig.tight_layout(pad=0.2)
    st.set_y(1.01)
    plt.show()
    plt.close(fig)
    plt.clf()
    path = './figures/matrix_pourbaix_T'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    fig.savefig(f'{path}/iter_{iter}_matrix_pourbaix.png',dpi=300, bbox_inches='tight')
    return cands, ids

def plot_surf_free_vs_T_contour(dfs, **kwargs):
    """fix U, plot scores vs. T. only change chem. pot. of Pd and Ti
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    U = kwargs['U']
    df_raw = kwargs['df_raw']
    Ts = sorted(set(df_raw['T']))
    minuss, idss = [], []
    for t in Ts:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_pd in d_mu_Pd:
            for d_mu_ti in d_mu_Ti:
                for pco in P_CO:
                    for u in U:
                        cand, id_set, minis, ids = plot_multi_scores_vs_T(dfs, d_mu_pd, d_mu_ti, pco, u, **kwargs)
                        xs.append(d_mu_pd)
                        ys.append(d_mu_ti)
                        zs.append(minis[t])
                        tags.append(ids[t])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(5,5), ys.reshape(5,5), zs.reshape(5,5))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(f'Iter. {iter}, U={U[0]} V, T={t} K, Pco={P_CO[0]} Pa', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        plt.show()
        path = './figures/SFE_T'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_T_{t}_SFE.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        plt.xlim([-3.25, -2.25])
        plt.ylim([-8.28, -7.28])
        # cs = get_colorslist(NUM_COLORS=1000)
        # colors = []
        # for tag in tags:
        #     colors.append(cs[tag])
        # plt.scatter(xs, ys, c=colors, marker='s',s=2800, linewidths=4)
        # plt.xlim([-3.37, -2.13])
        # plt.ylim([-8.41, -7.16])
        # for i, txt in enumerate(tags):
        #     ax.annotate(txt, (xs[i], ys[i]), ha='center')
        ax.set_title(f'Iter. {iter}, U={U[0]} V, T={t} K, Pco={P_CO[0]} Pa', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        plt.show()
        path = './figures/cands_T'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_T_{t}_cands.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()   
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss

def plot_surf_free_vs_T_matrix_all(dfs, **kwargs):
    """Generate len(T)xlen(P_CO) 5x5 matrix plot"""
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    P_CO = kwargs['P_CO']
    U = kwargs['U']
    iter = kwargs['iter']
    cands, ids = [], []
    path = './figures/matrix_all_T'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    path = f'./figures/matrix_all_T/iter_{iter}'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    for u in U:
        path = f'./figures/matrix_all_T/iter_{iter}/{u}'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        for pco in P_CO:
            fig = plt.figure(figsize=(12,10),dpi=300)
            i=0
            for d_mu_ti in d_mu_Ti:
                for d_mu_pd in d_mu_Pd:
                    ax = plt.subplot(5, 5, i+1)
                    cand, id_set, _, _ = plot_multi_scores_vs_T(dfs, d_mu_pd, d_mu_ti, pco, u, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 6
                    plt.text(0.05, 0.05, id_set, horizontalalignment='left', verticalalignment='center',
                             transform=ax.transAxes, fontsize=ft_sz)  
                    plt.text(0.05, 0.90, f'$\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}',
                             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,fontsize=ft_sz)
                    plt.xlim([283.15, 353.15])
                    plt.ylim([-1., 0.25])
                    i += 1
                    ft_sz = 10
                    if i % 5 == 1:
                        plt.ylabel('Surface free energy', fontsize=ft_sz)
                        plt.yticks([-1.00, -0.75, -0.50, -0.25, 0.00])
                    else:
                        plt.yticks([])
                    if i > 20:
                        plt.xlabel('Temperature (K)', fontsize=ft_sz)
                        # plt.xticks([-0.6, -0.4, -0.2, 0.0])
                    else:
                        plt.xticks([])
            st = plt.suptitle(f'Iter: {iter}, U={u} V, Pco={pco} Pa', fontsize=ft_sz)
            fig.tight_layout(pad=0.2)
            st.set_y(1.01)
            plt.show()
            fig.savefig(f'{path}/iter_{iter}_matrix_all_U_{u}_Pco_{pco}.png',dpi=300, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
    return cands, ids

# Pco
def plot_scores_Pco(df, i, d_mu_pd, d_mu_ti, t, u, x_col='P_CO', **kwargs):
    if 'df_raw' in kwargs:
        df_raw = kwargs['df_raw']
    kappa = df_raw['kappa'][0]
    df = df[(df['kappa']==kappa) & (df['d_mu_Pd']==d_mu_pd)  # change here
            & (df['d_mu_Ti']==d_mu_ti) & (df['U']==u) # x axis back and forward
            & (df['T']==t)]
    df = df.sort_values(x_col)
    x = df[x_col]
    y = -df['raw_scores']
    basic_plot(x, y, x_col, c=f"C{i}", lable=f'image{i}', ft_sz=12, **kwargs)
    return x, y

def plot_multi_scores_vs_Pco(dfs, d_mu_pd, d_mu_ti, t, u, x_col='P_CO', **kwargs):
    if 'df_raw' in kwargs:
        df_raw = kwargs['df_raw']
    P_CO = df_raw['P_CO']
    P_CO = list(sorted(set(P_CO)))
    results = pd.DataFrame(columns = P_CO)
    for i in range(len(dfs)):
        df = dfs[i]
        x, y = plot_scores_Pco(df, i, d_mu_pd, d_mu_ti, t, u, x_col='P_CO', **kwargs)
        assert (x.values == results.columns.values).all()
        results.loc[i] = y.values
    print('results:',results)
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
        if 'gray_above' in kwargs and kwargs['gray_above'] == True:
            if True:
                basic_plot(x, y, x_col, c=f'C{i}', lable=f'image{i}', ft_sz=12)
            else:
                cs = get_colorslist(NUM_COLORS=1000)
                basic_plot(x, y, x_col, c=cs[id], lable=f'image{i}', ft_sz=12)
    return cand, id_set, minis, ids

def plot_surf_free_vs_Pco(dfs, **kwargs):
    """fix U, plot scores vs. U"""
    if 'iter' in kwargs:
        iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    cands, ids = [], []
    path = './figures/pourbaix_Pco'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    # fig = plt.figure(dpi=300)
    for d_mu_ti in d_mu_Ti:
        for d_mu_pd in d_mu_Pd:
            for t in T:
                for u in U:
                    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                    cand, id_set, _, _ = plot_multi_scores_vs_Pco(dfs, d_mu_pd, d_mu_ti, t, u, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 12
                    plt.xlabel('Partial pressure of CO ($log_{10}(Pa)$)', fontsize=ft_sz)
                    plt.ylabel('Surface free energy (eV)', fontsize=ft_sz)
                    plt.xscale("log")
                    plt.xlim([0.101325, 101325])
                    plt.ylim([-1., 0.251])
                    plt.title(f'Iter {iter}, $\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}, U: {u} V, T: {t} K')
                    plt.text(0.34, 0.03, id_set, horizontalalignment='left', verticalalignment='center',
                              transform=ax.transAxes, fontsize=14, fontweight='bold')
                    path = f'./figures/pourbaix_Pco/Pd_{d_mu_pd}_Ti_{d_mu_ti}'
                    if not(os.path.exists(path) and os.path.isdir(path)):
                        os.mkdir(path)
                    fig.savefig(f'{path}/iter_{iter}_Pd_{d_mu_pd}_Ti_{d_mu_ti}.png',dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close(fig)
                    plt.clf()
    return cands, ids

def plot_surf_free_vs_Pco_matrix(dfs, **kwargs):
    """fix U, plot scores vs. U. only change chem. pot. of Pd and Ti"""
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    iter = kwargs['iter']
    cands, ids = [], []
    fig = plt.figure(figsize=(12,10),dpi=300)
    i=0
    for d_mu_ti in d_mu_Ti:
        for d_mu_pd in d_mu_Pd:
            for t in T:
                for u in U:
                    ax = plt.subplot(5, 5, i+1)
                    cand, id_set, _, _ = plot_multi_scores_vs_Pco(dfs, d_mu_pd, d_mu_ti, t, u, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 6
                    plt.text(0.05, 0.05, id_set, horizontalalignment='left', verticalalignment='center',
                             transform=ax.transAxes, fontsize=ft_sz)  
                    plt.text(0.05, 0.90, f'$\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}',
                             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,fontsize=ft_sz)
                    plt.xscale("log")
                    plt.xlim([0.101325, 101325])
                    plt.ylim([-1., 0.25])
                    i += 1
                    ft_sz = 10
                    if i % 5 == 1:
                        plt.ylabel('Surface free energy', fontsize=ft_sz)
                        plt.yticks([-1.00, -0.75, -0.50, -0.25, 0.00])
                    else:
                        plt.yticks([])
                    if i > 20:
                        plt.xlabel('Part. pressure of CO ($log_{10}(Pa)$)', fontsize=ft_sz)
                        # plt.xticks([-0.6, -0.4, -0.2, 0.0])
                    else:
                        plt.xticks([])
    st = plt.suptitle(f'Iter: {iter}, U={U[0]} V, T={T[0]} K', fontsize=ft_sz)
    fig.tight_layout(pad=0.2)
    st.set_y(1.01)
    plt.show()
    plt.close(fig)
    plt.clf()
    path = './figures/matrix_pourbaix_Pco'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    fig.savefig(f'{path}/iter_{iter}_matrix_pourbaix.png',dpi=300, bbox_inches='tight')
    return cands, ids

def plot_surf_free_vs_Pco_contour(dfs, **kwargs):
    """fix U, plot scores vs. T. only change chem. pot. of Pd and Ti
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    df_raw = kwargs['df_raw']
    Pcos = sorted(set(df_raw['P_CO']))
    minuss, idss = [], []
    for pco in Pcos:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_pd in d_mu_Pd:
            for d_mu_ti in d_mu_Ti:
                for t in T:
                    for u in U:
                        cand, id_set, minis, ids = plot_multi_scores_vs_Pco(dfs, d_mu_pd, d_mu_ti, t, u, **kwargs)
                        xs.append(d_mu_pd)
                        ys.append(d_mu_ti)
                        zs.append(minis[pco])
                        tags.append(ids[pco])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(5,5), ys.reshape(5,5), zs.reshape(5,5))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(f'Iter. {iter}, U={U[0]} V, T={T[0]} K, Pco={pco} Pa', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        plt.show()
        path = './figures/SFE_Pco'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_Pco_{pco}_SFE.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        # cs = get_colorslist(NUM_COLORS=1000)
        # colors = []
        # for tag in tags:
        #     colors.append(cs[tag])
        # plt.scatter(xs, ys, c=colors, marker='s',s=2800, linewidths=4)
        plt.xlim([-3.37, -2.13])
        plt.ylim([-8.41, -7.16])
        for i, txt in enumerate(tags):
            ax.annotate(txt, (xs[i], ys[i]), ha='center')
        ax.set_title(f'Iter. {iter}, U={U[0]} V, T={T[0]} K, Pco={pco} Pa', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        # plt.show()
        path = './figures/cands_Pco'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_Pco_{pco}_cands.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss

def plot_surf_free_vs_Pco_matrix_all(dfs, **kwargs):
    """Generate len(T)xlen(P_CO) 5x5 matrix plot"""
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    iter = kwargs['iter']
    cands, ids = [], []
    path = './figures/matrix_all_Pco'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    path = f'./figures/matrix_all_Pco/iter_{iter}'
    if not(os.path.exists(path) and os.path.isdir(path)):
        os.mkdir(path)
    for u in U:
        path = f'./figures/matrix_all_Pco/iter_{iter}/{u}'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        for t in T:
            fig = plt.figure(figsize=(12,10),dpi=300)
            i=0
            for d_mu_ti in d_mu_Ti:
                for d_mu_pd in d_mu_Pd:
                    ax = plt.subplot(5, 5, i+1)
                    cand, id_set, _, _ = plot_multi_scores_vs_Pco(dfs, d_mu_pd, d_mu_ti, t, u, **kwargs)
                    cands.append(cand)
                    ids.append(id_set)
                    ft_sz = 6
                    plt.text(0.05, 0.05, id_set, horizontalalignment='left', verticalalignment='center',
                             transform=ax.transAxes, fontsize=ft_sz)  
                    plt.text(0.05, 0.90, f'$\Delta \mu_{{Pd}}$: {d_mu_pd}, $\Delta \mu_{{Ti}}$: {round(d_mu_ti,3)}',
                             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,fontsize=ft_sz)
                    plt.xscale("log")
                    plt.xlim([0.101325, 101325])
                    plt.ylim([-1., 0.25])
                    i += 1
                    ft_sz = 10
                    if i % 5 == 1:
                        plt.ylabel('Surface free energy', fontsize=ft_sz)
                        plt.yticks([-1.00, -0.75, -0.50, -0.25, 0.00])
                    else:
                        plt.yticks([])
                    if i > 20:
                        plt.xlabel('Part. pressure of CO ($log_{10}(Pa)$)', fontsize=ft_sz)
                        # plt.xticks([-0.6, -0.4, -0.2, 0.0])
                    else:
                        plt.xticks([])
            st = plt.suptitle(f'Iter: {iter}, U={u} V, T={t} Pa', fontsize=ft_sz)
            fig.tight_layout(pad=0.2)
            st.set_y(1.01)
            plt.show()
            fig.savefig(f'{path}/iter_{iter}_matrix_all_U_{u}_T_{t}.png',dpi=300, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
    return cands, ids

def plot_heatmap(xs, ys, tags, df_results):
    """Plot heatmap of xs, ys, tags"""
    cs = get_colorslist(NUM_COLORS=1000)
    colors = []
    for tag in tags:
        colors.append(cs[tag])
    # im = ax.imshow(np.asarray(tags).reshape(9,5))
    xi = np.linspace(min(xs), max(xs), 100)
    yi = np.linspace(min(ys), max(ys), 100)
    from scipy.interpolate import griddata
    Z = griddata((xs, ys), np.asarray(tags), (xi[None,:], yi[:,None]), method='nearest')
    # plt.contour(xi, yi, Z)
    colors = []
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            colors.append(cs[Z[i][j]])
    X, Y = np.meshgrid(xi,yi)
    plt.scatter(X, Y, c=colors, marker='s',s=2, linewidths=4)
    pos_label = {}
    df = pd.DataFrame({'xs': X.flatten(), 'ys': Y.flatten(), 'zs': Z.flatten()})
    for tag in set(tags):
        df_sub = df.loc[df['zs'] == tag]
        x_ave = df_sub['xs'].mean()
        y_ave = df_sub['ys'].mean()
        pos_label[tag] = (x_ave, y_ave)
    return pos_label

def plot_heatmap_old2(xs, ys, tags, df_results):
    """Plot heatmap of xs, ys, tags"""
    cs = get_colorslist(NUM_COLORS=1000)
    colors = []
    for tag in tags:
        colors.append(cs[tag])
    # im = ax.imshow(np.asarray(tags).reshape(9,5))
    xi = np.linspace(min(xs), max(xs), 100)
    yi = np.linspace(min(ys), max(ys), 100)
    from scipy.interpolate import griddata
    Z = griddata((xs, ys), np.asarray(tags), (xi[None,:], yi[:,None]), method='nearest')
    # plt.contour(xi, yi, Z)
    colors = []
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            colors.append(cs[Z[i][j]])
    X, Y = np.meshgrid(xi,yi)
    plt.scatter(X, Y, c=colors, marker='s',s=2, linewidths=4)
    pos_label = {}
    for tag in set(tags):
        df_sub = df_results.loc[df_results['tags'] == tag]
        x_ave = df_sub['xs'].mean()
        y_ave = df_sub['ys'].mean()
        pos_label[tag] = (x_ave, y_ave)
    return pos_label

def plot_heatmap_old(xs, ys, tags):
    fig,ax = plt.subplots(figsize=(5,5), dpi=300)
    cs = get_colorslist(NUM_COLORS=1000)
    colors = []
    for tag in tags:
        colors.append(cs[tag])
    plt.scatter(xs, ys, c=colors, marker='s',s=2800, linewidths=4)
    # plt.xlim([-3.37, -2.13])
    # plt.ylim([-8.41, -7.16])
    for i, txt in enumerate(tags):
        ax.annotate(txt, (xs[i], ys[i]), ha='center')

def plot_surf_free_vs_T_and_Pco(dfs, **kwargs):
    """Fix chem. pot. of Pd and Ti and U. Plot T, Pco vs surf. energy
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    P_CO = kwargs['P_CO']
    minuss, idss = [], []
    for u in U:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_pd in d_mu_Pd:
            for d_mu_ti in d_mu_Ti:
                for t in T:
                    for pco in P_CO:
                        cand, id_set, minis, ids = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                        xs.append(t)
                        ys.append(np.log10(pco))
                        zs.append(minis[u])
                        tags.append(ids[u])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(5,5), ys.reshape(5,5), zs.reshape(5,5))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Pd}}$: {d_mu_Pd[0]}, $\Delta \mu_{{Ti}}$: {round(d_mu_Ti[0],3)} U={u} V', fontsize=ft_sz)
        plt.xlabel('Temperature (K)', fontsize=ft_sz)
        plt.ylabel('Partial pressure of CO ($log_{10}(Pa)$)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        plt.show()
        plt.close(fig)
        plt.clf()
        path = './figures/Contour_vs_T_and_Pco'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_U_{u}_Contour.png',dpi=300, bbox_inches='tight')
        
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Pd}}$: {d_mu_Pd[0]}, $\Delta \mu_{{Ti}}$: {round(d_mu_Ti[0],3)} U={u} V', fontsize=ft_sz)
        plt.xlim([283.15, 353.15])
        plt.ylim([-1, 5])
        plt.xlabel('Temperature (K)', fontsize=ft_sz)
        plt.ylabel('Partial pressure of CO ($log_{10}(Pa)$)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        plt.show()
        path = './figures/Heatmap_vs_T_and_Pco'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_U_{u}_heatmap.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss

def plot_surf_free_vs_U_and_Pco(dfs, **kwargs):
    """Fix chem. pot. of Pd and Ti and T. Plot U, Pco vs surf. energy
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    P_CO = kwargs['P_CO']
    minuss, idss = [], []
    for t in T:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_pd in d_mu_Pd:
            for d_mu_ti in d_mu_Ti:
                for u in U:
                    for pco in P_CO:
                        cand, id_set, minis, ids = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                        xs.append(u)
                        ys.append(np.log10(pco))
                        zs.append(minis[u])
                        tags.append(ids[u])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(9,5), ys.reshape(9,5), zs.reshape(9,5))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Pd}}$: {d_mu_Pd[0]}, $\Delta \mu_{{Ti}}$: {round(d_mu_Ti[0],3)} T={T[0]} K', fontsize=ft_sz)
        plt.xlabel('Potential (V)', fontsize=ft_sz)
        plt.ylabel('Partial pressure of CO ($log_{10}(Pa)$)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        plt.show()
        plt.close(fig)
        plt.clf()
        path = './figures/Contour_vs_U_and_Pco'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_T_{t}_Contour.png',dpi=300, bbox_inches='tight')
        
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Pd}}$: {d_mu_Pd[0]}, $\Delta \mu_{{Ti}}$: {round(d_mu_Ti[0],3)} T={T[0]} K', fontsize=ft_sz)
        plt.xlim([-0.8, 0.0])
        plt.ylim([-1, 5])
        plt.xlabel('Potential (V)', fontsize=ft_sz)
        plt.ylabel('Partial pressure of CO ($log_{10}(Pa)$)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        plt.show()
        path = './figures/Heatmap_vs_U_and_Pco'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_T_{t}_heatmap.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss

def plot_surf_free_vs_T_and_U(dfs, **kwargs):
    """Fix chem. pot. of Pd and Ti and Pco. Plot T, U vs surf. energy
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    P_CO = kwargs['P_CO']
    minuss, idss = [], []
    for pco in P_CO:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_pd in d_mu_Pd:
            for d_mu_ti in d_mu_Ti:
                for u in U:
                    for t in T:
                        cand, id_set, minis, ids = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                        xs.append(u)
                        ys.append(t)
                        zs.append(minis[u])
                        tags.append(ids[u])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(9,5), ys.reshape(9,5), zs.reshape(9,5))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Pd}}$: {d_mu_Pd[0]}, $\Delta \mu_{{Ti}}$: {round(d_mu_Ti[0],3)} Pco={pco} Pa', fontsize=ft_sz)
        plt.xlabel('Potential (V)', fontsize=ft_sz)
        plt.ylabel('Temperature (K)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        plt.show()
        plt.close(fig)
        plt.clf()
        path = './figures/Contour_vs_T_and_U'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_pot_{U}_Contour.png',dpi=300, bbox_inches='tight')
        
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Pd}}$: {d_mu_Pd[0]}, $\Delta \mu_{{Ti}}$: {round(d_mu_Ti[0],3)} Pco={pco} Pa', fontsize=ft_sz)
        plt.xlim([-0.8, 0.0])
        plt.ylim([283.15, 353.15])
        plt.xlabel('Potential (V)', fontsize=ft_sz)
        plt.ylabel('Temperature (K)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        plt.show()
        path = './figures/Heatmap_vs_T_and_U'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_Pco_{pco}_heatmap.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss

def plot_surf_free_vs_U_and_Tichem(dfs, **kwargs):
    """Fix chem. pot. of Pd, and Pco. Plot U, Tichem vs surf. energy
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    P_CO = kwargs['P_CO']
    minuss, idss = [], []
    for d_mu_pd in d_mu_Pd:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_ti in d_mu_Ti:
            for pco in P_CO:
                for u in U:
                    for t in T:
                        cand, id_set, minis, ids = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                        xs.append(d_mu_ti)
                        ys.append(u)
                        zs.append(minis[u])
                        tags.append(ids[u])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(5,9), ys.reshape(5,9), zs.reshape(5,9))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(rf'Iter. {iter}, $\Delta \mu_{{Pd}}$: {d_mu_pd}, Pco={pco} Pa, T={T[0]} K', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.ylabel('Potential (V)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        plt.show()
        plt.close(fig)
        plt.clf()
        path = './figures/Contour_vs_Tichem_and_U'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_pot_{U}_Contour.png',dpi=300, bbox_inches='tight')
        
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Pd}}$: {d_mu_pd}, Pco={pco} Pa, T={T[0]} K', fontsize=ft_sz)
        plt.xlim([-8.28, -7.28])
        plt.ylim([-0.8, 0.0])
        plt.xlabel(r'$\Delta \mu_{Ti}$', fontsize=ft_sz)
        plt.ylabel('Potential (V)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        plt.show()
        path = './figures/Heatmap_vs_Tichem_and_U'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_Pco_{pco}_heatmap.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss

def plot_surf_free_vs_U_and_Pdchem(dfs, **kwargs):
    """Fix chem. pot. of Ti, and Pco. Plot U, Pdchem vs surf. energy
    Generat only one 5x5 matrix plot"""
    iter = kwargs['iter']
    d_mu_Pd = kwargs['d_mu_Pd']
    d_mu_Ti = kwargs['d_mu_Ti']
    T = kwargs['T']
    U = kwargs['U']
    P_CO = kwargs['P_CO']
    minuss, idss = [], []
    for d_mu_ti in d_mu_Ti:
        i=0
        xs, ys, zs, tags = [], [], [], []
        for d_mu_pd in d_mu_Pd:
            for pco in P_CO:
                for u in U:
                    for t in T:
                        cand, id_set, minis, ids = plot_multi_scores_vs_U(dfs, d_mu_pd, d_mu_ti, pco, t, **kwargs)
                        xs.append(d_mu_pd)
                        ys.append(u)
                        zs.append(minis[u])
                        tags.append(ids[u])
        df_results = pd.DataFrame({'xs': xs, 'ys': ys, 'zs': zs, 'tags':tags})
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)
        fig,ax = plt.subplots(figsize=(8,6), dpi=300)
        ft_sz = 12
        cp = ax.contourf(xs.reshape(5,9), ys.reshape(5,9), zs.reshape(5,9))
        bar = fig.colorbar(cp) # Add a colorbar to a plot
        bar.ax.invert_yaxis()
        bar.set_label('Surface free energy (eV)', rotation=270, labelpad=20, fontsize=ft_sz)
        bar.ax.tick_params(labelsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Ti}}$: {d_mu_ti}, Pco={pco} Pa, T={T[0]} K', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel('Potential (V)', fontsize=ft_sz)
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        if False:
            plt.scatter(xs, ys, c='gray', marker='o',)
            for i, txt in enumerate(tags):
                ax.annotate(txt, (xs[i], ys[i]), ha='center')
        plt.show()
        plt.close(fig)
        plt.clf()
        path = './figures/Contour_vs_Tichem_and_U'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_pot_{U}_Contour.png',dpi=300, bbox_inches='tight')
        
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ft_sz = 10
        pos_label = plot_heatmap(xs, ys, tags, df_results)
        for label in pos_label.keys():
            x, y = pos_label[label]
            plt.text(x, y, label, horizontalalignment='center', fontsize=ft_sz)
        ax.set_title(f'Iter. {iter}, $\Delta \mu_{{Ti}}$: {d_mu_ti}, Pco={pco} Pa, T={T[0]} K', fontsize=ft_sz)
        plt.xlabel(r'$\Delta \mu_{Pd}$', fontsize=ft_sz)
        plt.ylabel('Potential (V)', fontsize=ft_sz)
        plt.xlim([-3.25, -2.25])
        plt.ylim([-0.8, 0.0])
        plt.xticks(fontsize=ft_sz)
        plt.yticks(fontsize=ft_sz)
        plt.show()
        path = './figures/Heatmap_vs_Tichem_and_U'
        if not(os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
        fig.savefig(f'{path}/iter_{iter}_Pco_{pco}_heatmap.png',dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.clf()
        minuss.append(minis)
        idss.append(ids)
    return minuss, idss

if __name__ == '__main__':
    ''

    
