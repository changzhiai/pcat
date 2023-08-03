# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:30:44 2023

@author: changai
"""
import matplotlib.pyplot as plt 
from pcat.lib.io import pd_read_excel
import numpy as np
import matplotlib as mpl
import pickle
from pcat.free_energy import HERFED
from pcat.free_energy import CO2RRFED
from pcat.scaling_relation import ScalingRelation
from pcat.selectivity import Selectivity

def plot_CO2RR_free_energy_old():
    """New version to analyze data by pandas"""
    from pcat.free_energy import CO2RRFEDplot
    df = pd_read_excel(filename='./data/iter25.xlsx', sheet='free_energy')
    ColorDict = {'image0': 'C0', 'image1': 'C1', 'image2': 'C2', 'image3': 'C3', 'image4': 'C4', }
    stepsNames = ['* + CO$_{2}$', 'HOCO*', 'CO*', '* + CO']
    CO2RR_FED = CO2RRFEDplot(stepsNames, df.index,df.values, fig_name='./CO2RR_FED.jpg', ColorDict=ColorDict)
    CO2RR_FED.plot(save=True, title='')
    
def plot_activity():
    from pcat.activity import Activity
    """Plot activity of CO2RR"""
    df = pd_read_excel(filename='./data/iter31.xlsx', sheet='Activity')
    # df.drop(['Pd16Ti48H8', 'Pd16Ti48H24'], inplace=True)
    color = ['black', 'blue']
    ColorDict= {'Pure': color[1], 'Pd14Ti2H17+1CO': color[1], 'Pd5Ti11H20+2CO': color[1], 
                'Pd5Ti11H20+1CO': color[1], 'Pd9Ti7H17+1CO': color[1]}
    # tune_tex_pos = {'Pure': [-0.0, 0.0], 'Pd14Ti2H17+1CO': [-0.25, -0.17], 'Pd5Ti11H20+2CO': [-0.22, -0.28], 
    #                 'Pd5Ti11H20+1CO': [-0.25, -0.25], 'Pd9Ti7H17+1CO': [-0.25, -0.25]}
    tune_tex_pos = {'Pure': [-0.0, 0.0], 'Pd14Ti2H17+1CO': [-0.2, -0.17], 'Pd5Ti11H20+2CO': [-0.12, -0.28], 
                    'Pd5Ti11H20+1CO': [-0.2, -0.25], 'Pd9Ti7H17+1CO': [-0.15, -0.25]}
    name_fig_act = './figures/iter31_activity.jpg'
    activity = Activity(df, descriper1 = 'E(CO*)', descriper2 = 'E(HOCO*)', fig_name=name_fig_act,
                        U0=-0.5, 
                        T0=297.15, 
                        pCO2g = 1., 
                        pCOg=0.005562, 
                        pH2Og = 1., 
                        cHp0 = 10.**(-0.),
                        Gact=0.2, 
                        p_factor = 3.6 * 10**4)
    # activity.verify_BE2FE()
    # activity.plot(save=True, text=False, tune_tex_pos=tune_tex_pos, ColorDict=ColorDict)
    # activity.plot(save=True, xlim=[-2.5, 2.5], ylim=[-2.5, 2.5])
    activity.plot(save=True, text=False, tune_tex_pos=tune_tex_pos, ColorDict=ColorDict, xlim=[-1.0, 0.35], ylim=[-1.2, 1.0], **{'subscritpt': True})
    
def plot_CO2RR_free_enegy():
    """
    Plot free energy for CO2RR
    """
    df = pd_read_excel(filename='./data/iter31.xlsx', sheet='CO2RR_FED')
    # obj_list = ['Pd64H64',]
    # df = df[df.index.isin(obj_list)]
    step_names = ['* + CO$_{2}$', 'HOCO*', 'CO*', '* + CO']  #reload step name for CO2RR
    df.set_axis(step_names, axis='columns', inplace=True)
    name_fig_FE = './figures/iter31_CO2RR.jpg'
    fig = plt.figure(figsize=(8, 6), dpi = 300)
    ax = fig.add_subplot(111)
    ColorDict= {'Pure': 'black',}
    CO2RR_FED = CO2RRFED(df, fig_name=name_fig_FE, **{'ColorDict': ColorDict})
    pos0, _ = CO2RR_FED.plot(ax=ax, save=False, title='', **{'subscritpt': True})
    print('initial x pos:', pos0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.46, -0.12), fancybox=True, shadow=True, ncol=5, fontsize=8)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.12), fancybox=True, shadow=True, ncol=5, fontsize=8)
    # plt.legend(loc = "lower left", bbox_to_anchor=(0.00, -0.50, 0.8, 1.02), ncol=5, borderaxespad=0)
    plt.show()
    fig.savefig(name_fig_FE, dpi=300, bbox_inches='tight')
    
def plot_HER_free_energy():
    """Plot free energy for HER"""
    df = pd_read_excel(filename='./data/iter31.xlsx', sheet='HER_FED')
    df['step1']=0
    df['step3']=0
    df = df[['step1', 'G(H*)', 'step3']]
    step_names = ['* + $H^{+}$', 'H*', r'* + $\frac{1}{2}H_{2}$']
    df.set_axis(step_names, axis='columns', inplace=True)
    # obj_list = ['Pd64H64',]
    # df = df[df.index.isin(obj_list)]
    name_fig_FE = './figures/iter31_HER.jpg'
    fig = plt.figure(figsize=(8, 6), dpi = 300)
    ax = fig.add_subplot(111)
    ColorDict= {'Pure': 'black',}
    HER_FED = HERFED(df, fig_name=name_fig_FE, **{'ColorDict': ColorDict})
    pos0, _ = HER_FED.plot(ax=ax, save=False, title='', **{'subscritpt': True})
    print('initial x pos:', pos0)
    # plt.legend(loc='upper center', bbox_to_anchor=(pos0-0.25, -0.12), fancybox=True, shadow=True, ncol=5, fontsize=8)
    # plt.legend(loc='upper center', bbox_to_anchor=(pos0-1.96, -0.12), fancybox=True, shadow=True, ncol=5, fontsize=8)
    plt.legend(loc='upper center', bbox_to_anchor=(pos0-0.75, -0.12), fancybox=True, shadow=True, ncol=5, fontsize=8)
    plt.show()
    fig.savefig(name_fig_FE, dpi=300, bbox_inches='tight')
    
def plot_selectivity():
    """Plot selectivity of CO2RR and HER"""
    
    df = pd_read_excel(filename='./data/iter31.xlsx', sheet='Selectivity')
    df = df[['G_HOCO-G_H']]
    # df.set_axis(['Single'], axis='columns', inplace=True)
    name_fig_select = './figures/iter31_Selectivity.jpg'
    
    tune_ano_pos = {'Pure': [0.7, 0.1], 'Pd14Ti2H17+1CO': [0.7, 0.0], 'Pd5Ti11H20+2CO': [0.7, -0.1], 
                    'Pd5Ti11H20+1CO': [0.7, -0.1], 'Pd9Ti7H17+1CO': [0.7, 0.0]}
    selectivity = Selectivity(df, fig_name=name_fig_select)
    selectivity.plot(save=True, title='', xlabel='Different surfaces', tune_tex_pos=-0.4, legend=False, 
                     tune_ano_pos=tune_ano_pos, **{'width': 0.5, 'head_width': 1., 'subscritpt': True})

def plot_iteration():
    mpl.rcParams["figure.figsize"] = [6.4, 4.8]
    fig, ax = plt.subplots(dpi=300)
    # fig, ax = plt.subplots(figsize=(12,10), dpi=300)
    # fig = plt.figure(figsize=(12,10),dpi=300)
    # lens = [103, 129, 154, 180, 209, 236, 265, 284, 308, 328,
    #         345, 371, 392, 410, 429, 458, 499, 575, 634, 693,
    #         742, 775, 796, 819, 840, 865, 882, 913, 933, 955,
    #         976]
    iters = ['iter1', 'iter2','iter3','iter4','iter5','iter6','iter7','iter8','iter9','iter10',
             'iter11','iter12','iter13','iter14','iter15','iter16','iter17','iter18','iter19','iter20',
             'iter21','iter22','iter23','iter24','iter25','iter26','iter27','iter28','iter29','iter30',
             'iter31']
    rmse_e = [41.039, 31.736, 29.180, 27.135, 25.770, 23.888, 21.107, 22.015, 19.886, 21.128,
              20.094, 19.295, 18.923, 17.493, 18.186, 16.965, 16.166, 17.255, 15.546, 15.202, 
              14.569, 14.216, 12.739, 14.475, 13.698, 13.492, 12.066, 13.367, 12.777, 12.439,
              12.425]
    rmse_e3 = [1.306, 5.019, 8.416, 7.626, 7.365, 6.086, 6.382, 6.130, 5.871, 5.563,
               5.521, 5.805, 5.541, 5.708, 5.575, 5.566, 4.971, 6.751, 7.095, 6.136,
               5.346, 5.547, 5.913, 5.726, 5.277, 5.057, 5.202, 5.398, 5.636, 5.431,
               5.267]
    iters = np.arange(1, len(iters)+1, 1)
    plt.plot(iters, rmse_e, '-o', label='The active learning with all iterations')
    plt.plot(iters[1:], rmse_e3[1:], '-o', label='The active learning without 1st iteration')
    # plt.ylim([3, 50])
    plt.xlabel('The number of iterations')
    plt.ylabel('Average energy RMSE (meV/atom)')
    plt.legend()
    plt.show()
    fig.savefig('./figures/RMSE_iters.png', dpi=300, bbox_inches='tight')
    print(fig.get_figwidth(), fig.get_figheight())
    
def read_list(iteration, name='all_ids.pkl'):
    with open(f'./figures/matrix_all/iter_{iteration}/{name}', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def flatten_list(n_list):
    import itertools
    n_list = list(itertools.chain(*n_list))
    n_list = list(set(n_list))
    return n_list
    
def find_num_of_new(pre_unique_ids, unique_ids):
    new_ids = []
    for id in unique_ids:
        if id not in pre_unique_ids:
            new_ids.append(id)
    return new_ids

def new_cands_vs_iters(since=1, iter=32):
    iter_unique_ids, iter_new_ids = [], []
    iter_unique_ids.append([])
    iter_new_ids.append([])
    for i in range(1,iter):
        print(i)
        unique_ids = read_list(i, name='all_unique_ids.pkl')
        iter_unique_ids.append(unique_ids)
        new_ids = find_num_of_new(iter_unique_ids[i-1], iter_unique_ids[i])
        iter_new_ids.append(new_ids)
        print(sorted(unique_ids))
    # fig, ax = plt.subplots(figsize=(8,7), dpi=300)
    fig, ax = plt.subplots(dpi=300)
    lens_new = []
    for new_ids in iter_new_ids:
        lens_new.append(len(new_ids))
    print(lens_new)
    ft_sz = 12
    xs = np.arange(0, iter, 1)[since:]
    ys = lens_new[since:]
    plt.plot(xs, ys, '-o')
    plt.xlabel('The numbe of iterations', fontsize=ft_sz)
    plt.ylabel('The number of new candidates', fontsize=ft_sz)
    plt.xticks(np.arange(since, iter, 1),fontsize=ft_sz)
    plt.yticks(fontsize=ft_sz)
    plt.show()

if __name__ == '__main__':
    if False:
        plot_activity()
        plot_HER_free_energy()
        plot_CO2RR_free_enegy()
        plot_selectivity()
    if True:
        plot_iteration()
        new_cands_vs_iters(since=17)
    
