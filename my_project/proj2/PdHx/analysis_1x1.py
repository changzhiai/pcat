# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:15:03 2022

@author: changai
"""

from ase.db import connect
import matplotlib.pyplot as plt
import numpy as np

system_name = 'collect_vasp_1x1_PdHx'
db_name = f'./data/{system_name}.db'
xls_name = f'./data/{system_name}.xlsx'
fig_dir = './figures'
db = connect(db_name)
energies = []
for row in db.select():
    formula = row.formula
    energy = row.energy
    energies.append(energy)

fig = plt.figure(dpi=300)
plt.plot(np.arange(1, 10, 1), energies[1:], '-o')
plt.xticks(np.arange(1, 10, 1))
plt.xlabel('Remove ith layer from bottom to top')
plt.ylabel('DFT energy (eV)')
plt.show()