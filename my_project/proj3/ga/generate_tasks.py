# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:33:46 2022

@author: changai
"""

from ase.units import kB
from scipy.optimize import bisect
import numpy as np
import pandas as pd

# T=298.15, U=0, pH=0, P_H2=101325., P_CO2=101325., P_H2O=3534., P_CO=5562.

def get_conditions():
    """Get reaction conditions"""
    T = np.array([298.15, 398.15])
    U = np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8])
    pH = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    P_H2 = np.array([101325.])
    P_CO2 = np.array([101325.])
    P_H2O = np.array([3534.])
    P_CO = np.array([5562.])
    kappa = np.array([0, 1, 2, 3, 4])
    
    cond_grid = np.array(np.meshgrid(T, U, pH, P_H2, P_CO2, P_H2O, P_CO, kappa)).T.reshape(-1, 8)
    
    df = pd.DataFrame(cond_grid, columns=['T', 'U', 'pH', 'P_H2', 'P_CO2', 'P_H2O', 'P_CO', 'kappa'])
    df.to_pickle('em_tasks.pkl')
    df.to_csv('em_tasks.csv')
    return df

# get_conditions()
niches = pd.read_pickle('./em_tasks.pkl')