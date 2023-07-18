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

def free_energy_pd():
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
    df = pd_read_excel(filename='./data/iter25.xlsx', sheet='binding_energy')
    # df.drop(['Pd16Ti48H8', 'Pd16Ti48H24'], inplace=True)
    ColorDict= {'images_774_0': 'white',}
    
    tune_tex_pos = {'images_774_0': [-0.5, -0.0],}
    name_fig_act = './data/iter25_activity.jpg'
    activity = Activity(df, descriper1 = 'E(*CO)', descriper2 = 'E(*HOCO)', fig_name=name_fig_act,
                        U0=-0.5, 
                        T0=297.15, 
                        pCO2g = 1., 
                        pCOg=0.005562, 
                        pH2Og = 1., 
                        cHp0 = 10.**(-0.),
                        Gact=0.2, 
                        p_factor = 3.6 * 10**4)
    # activity.verify_BE2FE()
    activity.plot(save=True, text=True, tune_tex_pos=tune_tex_pos, ColorDict=ColorDict)
    # activity.plot(save=True,)
    # activity.plot(save=True, xlim=[-2.5, 2.5], ylim=[-2.5, 2.5])
    # activity.plot(save=True, xlim=[-1., 0], ylim=[-0., 1])
    # activity.plot(save=True, xlim=[-2.5, 1.0], ylim=[-2.5, 1])

def plot_iteration():
    mpl.rcParams["figure.figsize"] = [6.4, 4.8]
    fig, ax = plt.subplots(dpi=300)
    # fig, ax = plt.subplots(figsize=(12,10), dpi=300)
    # fig = plt.figure(figsize=(12,10),dpi=300)
    iters = ['iter1', 'iter2','iter3','iter4','iter5','iter6','iter7','iter8','iter9','iter10',
             'iter11','iter12','iter13','iter14','iter15','iter16','iter17', 'iter18', 'iter19', 'iter20',
             'iter21', 'iter22', 'iter23', 'iter24', 'iter25', 'iter26', 'iter27']
    # lens = [103, 129, 154, 180, 209, 236, 265, 284, 308, 328,
    #         345, 371, 392, 410, 429, 458, 499, 575, 634, 693,
    #         742, 775, 796, 819, 840, 865, 882]
    rmse_e = [41.039, 31.736, 29.180, 27.135, 25.770, 23.888, 21.107, 22.015, 19.886, 21.128,
              20.094, 19.295, 18.923, 17.493, 18.186, 16.965, 16.166, 17.255, 15.546, 15.202, 
              14.569, 14.216, 12.739, 14.475, 13.698, 13.492, 12.066]

    rmse_e3 = [1.306, 5.019, 8.416, 7.626, 7.365, 6.086, 6.382, 6.130, 5.871, 5.563,
               5.521, 5.805, 5.541, 5.708, 5.575, 5.566, 4.971, 6.751, 7.095, 6.136,
               5.346, 5.547, 5.913, 5.726, 5.277, 5.057, 5.202]
    iters = np.arange(1, len(iters)+1, 1)
    # plt.plot(lens, rmse_e, '-o', label='Totol active learning')
    # plt.plot(lens[1:], rmse_e3[1:], '-o', label='Remove first generation')
    plt.plot(iters, rmse_e, '-o', label='The active learning with all iterations')
    plt.plot(iters[1:], rmse_e3[1:], '-o', label='The active learning without 1st iteration')
    # plt.ylim([3, 50])
    # plt.xlabel('The number of structures')
    plt.xlabel('The number of iterations')
    plt.ylabel('Average energy RMSE (meV/atom)')
    plt.legend()
    plt.show()
    # fig.savefig('RMSE_iters.png', dpi=300, bbox_inches='tight')
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

def new_cands_vs_iters(since=1, iter=28):
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
    fig, ax = plt.subplots(figsize=(8,7), dpi=300)
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
    plt.xticks(fontsize=ft_sz)
    plt.yticks(fontsize=ft_sz)
    plt.show()

if __name__ == '__main__':
    # free_energy_pd()
    # plot_activity()
    # plot_iteration()
    new_cands_vs_iters(since=17)
