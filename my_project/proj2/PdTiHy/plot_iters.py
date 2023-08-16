# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 21:59:08 2023

@author: changai
"""

import matplotlib.pyplot as plt 
import numpy as np

fig = plt.figure(dpi=300)
# fig, ax = plt.subplots(dpi=300)
# iters = ['iter1', 'iter2','iter3','iter4','iter5','iter6','iter7','iter8','iter9','iter10',
#          'iter11','iter12']
iters = list(np.arange(1, 13, 1))
cv10_Ti = [10.276, 3.572, 3.575, 3.654, 3.499, 3.654, 2.446, 2.287, 2.199, 2.093, 2.091, 2.011]
rmse_Ti = [0.481, 1.405, 2.022, 2.215, 2.421, 2.139, 2.139, 2.063, 1.989, 1.956, 1.939, 1.905]
plt.plot(iters, cv10_Ti, '-o', label='10-fold CV', color='C2')
plt.plot(iters, rmse_Ti, '-o', label='RMSE', color='C1')
plt.xticks(np.arange(1, 13, step=1))
plt.xlabel('The number of iterations')
plt.ylabel('Error (meV/atom)')
plt.legend()
plt.show()
fig.savefig('./figures/error_Ti.png', dpi=300, bbox_inches='tight')


fig = plt.figure(dpi=300)
iters = list(np.arange(1, 13, 1))
cv10_Nb = [12.500, 3.103, 3.234, 3.354, 3.370, 3.348, 3.350, 3.278, 3.274, 3.320, 3.274, 3.226]
rmse_Nb = [0.773, 2.508, 2.934, 3.176, 3.254, 3.259, 3.237, 3.203, 3.149, 3.234, 3.202, 3.157]
plt.plot(iters, cv10_Nb, '-o', label='10-fold CV', color='C2')
plt.plot(iters, rmse_Nb, '-o', label='RMSE', color='C1')
plt.xticks(np.arange(1, 13, step=1))
plt.xlabel('The number of iterations')
plt.ylabel('Error (meV/atom)')
plt.legend()
plt.show()
fig.savefig('./figures/error_Nb.png', dpi=300, bbox_inches='tight')