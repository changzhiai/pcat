# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:19:47 2023

@author: changai
"""
import matplotlib.pyplot as plt 

fig, ax = plt.subplots(dpi=300)
iters = ['iter1', 'iter2','iter3','iter4','iter5','iter6','iter7','iter8','iter9','iter10',
         'iter11','iter12','iter13','iter14','iter15','iter16','iter17', 'iter18', 'iter19', 'iter20',
         'iter21', 'iter22', 'iter23', 'iter24', 'iter25']
lens = [103, 129, 154, 180, 209, 236, 265, 284, 308, 328,
        345, 371, 392, 410, 429, 458, 499, 575, 634, 693,
        742, 775, 796, 819, 840]
rmse_e = [41.039, 31.736, 29.180, 27.135, 25.770, 23.888, 21.107, 22.015, 19.886, 21.128,
          20.094, 19.295, 18.923, 17.493, 18.186, 16.965, 16.166, 17.255, 15.546, 15.202, 
          14.569, 14.216, 12.739, 14.475, 13.698]

rmse_e2 = [14.502, 14.805, 12.502, 10.925, 9.950, 9.260, 8.461, 8.604, 9.058, 8.236, 
          5.626, 7.899, 8.403, 7.395, 8.675, 6.933, 6.858] # remove 40

rmse_e3 = [1.306, 5.019, 8.416, 7.626, 7.365, 6.086, 6.382, 6.130, 5.871, 5.563,
           5.521, 5.805, 5.541, 5.708, 5.575, 5.566, 4.971, 6.751, 7.095, 6.136,
           5.346, 5.547, 5.913, 5.726, 5.277]
plt.plot(lens, rmse_e, '-o', label='Totol active learning')
plt.plot(lens[1:], rmse_e3[1:], '-o', label='Remove first generation')
plt.ylim([0, 50])
plt.xlabel('The number of structures')
plt.ylabel('Energy RMSE (meV/atom)')
plt.legend()
plt.show()
# fig.savefig('RMSE_gens.png', dpi=300)