# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:54:15 2022

@author: changai
"""

import sys
sys.path.append("../../..")

from plotpackage.lib.io import read_excel, read_csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import matplotlib
plt.rcParams.update({'mathtext.default':  'regular', 'figure.dpi': 300, 'font.size': 12})
# plt.rcParams['figure.constrained_layout.use'] = True

xls_name = 'sites.xlsx'
fig_dir = 'bonds'
sheet = 'bonds' #Sheet1 by defaut


df = pd.read_excel(xls_name, sheet_name=sheet, index_col=0, header=0)
# x = df.index.values
# y = df['Single'].values
# plt.bar(x, y, facecolor='blue', alpha=0.5)
# plt.axhline(y=2.7,linewidth=1, color='r')
# plt.xlabel('Doped elements')
# plt.ylabel('O-M bond lengh')
# plt.show()
text = ['Single', 'Dimer', 'Triangle', 'Parall.', 'Island', 'Overlayer']
fig = plt.figure(figsize=(12, 12), dpi = 300)
# fig = plt.figure(figsize=(14, 18), dpi = 300)
i = 0
for conf in df.columns.values:
    ax = plt.subplot(3, 2, i + 1)
    x = df.index.values
    y = df[conf].values
    plt.bar(x, y, facecolor='blue', alpha=0.5)
    plt.axhline(y=2.7,linewidth=1, color='red')
    ax.text(-0.13, 0.97, string.ascii_lowercase[i], transform=ax.transAxes, size=20, weight='bold')
    plt.text(0.05, 0.93, text[i], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='black', fontweight='bold')
    # plt.xlabel('Doped elements')
    # plt.ylabel('O-M bond lengh')
    plt.ylim((0, 3.5))
    for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2) #linewith of frame
    # plt.title(conf)
    if i==4 or i==5:
        plt.xlabel('Doped elements', fontsize=14,)
    if i==0 or i==2 or i==4:
        plt.ylabel('O-M bond lengh', fontsize=14,)
    i += 1
plt.show()
fig.savefig(fig_dir+'/bonds.png', dpi=300, bbox_inches='tight')