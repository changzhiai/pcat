# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 09:54:00 2022

@author: changai
"""

from convex_hull_ce_PdHx import plot_convex_hull_PdHx as plot_convex_hull_PdHx_ce
from convex_hull_ce_PdHx import db2xls_ce
from convex_hull_dft_PdHx import plot_convex_hull_PdHx_dft
from convex_hull_dft_PdHx import db2xls_dft

for i in [8]:
    system_dft = f'PdHx_train_r{i}' # round 1, ce and dft
    # system_ce = f'results_r{i}'
    system_ce = f'PdHx_train_r{i}'

    fig_dir = './figures/'
    data_dir = './data'
    db_name_dft = f'./{data_dir}/{system_dft}.db'
    xls_name_dft = f'./{data_dir}/{system_dft}.xlsx'
    db_name_ce = f'./{data_dir}/{system_ce}.db'
    xls_name_ce = f'./{data_dir}/{system_ce}_ce.xlsx'
    
    sheet_name_convex_hull = 'convex_hull'

    if False:
        # db2xls_dft(db_name_dft)
        db2xls_ce(db_name_ce, xls_name_ce, sheet_name_convex_hull)
        print('excel generating done!')
    if True:
        fig, ax = plot_convex_hull_PdHx_dft(db_name_dft, cand=False, round=i, xls_name=xls_name_dft, sheet_name_convex_hull=sheet_name_convex_hull)
        plot_convex_hull_PdHx_ce(db_name_ce, cand=False, round=i, xls_name=xls_name_ce, sheet_name_convex_hull=sheet_name_convex_hull, fig=fig, ax=ax)
    # plot_convex_hull_PdHx_ce(db_name_ce, cand=False, round=i, xls_name=xls_name_ce, sheet_name_convex_hull=sheet_name_convex_hull)