# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:15:03 2022

@author: changai
"""

from ase.db import connect
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
# system_name = 'collect_vasp_1x1_PdHx'
# system_name = 'collect_vasp_1x1_PdHx_fix2b'
# system_name = 'collect_vasp_1x1_PdHx_fix2b_20v' # 1x1
system_name = 'collect_vasp_rm_1layers_H' # 4x4
db_name = f'./data/{system_name}.db'
# view(db_name)
xls_name = f'./data/{system_name}.xlsx'
fig_dir = './figures'
db = connect(db_name)
energies = []
for row in db.select():
    formula = row.formula
    energy = row.energy
    energies.append(energy)

fig = plt.figure(dpi=300)
# plt.plot(energies, '-o')
# es = energies[1:]
# es.reverse()
# plt.plot(es, '-o')
plt.plot(np.arange(1, len(energies), 1), energies[1:], '-or')
plt.xticks(np.arange(1, len(energies), 1))
# plt.xlabel('Remove ith layer from bottom to top')
plt.xlabel('Remove ith layer')
plt.ylabel('DFT energy (eV)')
plt.show()


system_name = 'collect_vasp_1x1_PdHx_fix2b_20v' # 1x1
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
es = energies[1:]
es.reverse()
plt.plot(np.arange(1, len(energies), 1), es, '-or')
plt.xticks(np.arange(1, len(energies), 1))
# plt.xlabel('Remove ith layer from bottom to top')
plt.xlabel('Remove ith layer')
plt.ylabel('DFT energy (eV)')
plt.show()