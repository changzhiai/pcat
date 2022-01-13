# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:44:15 2022

@author: changai
"""
kB = 8.617e-5 # Boltzmann constant in eV/K
hplanck = 4.135669e-15 # Planck constant in eV s

E_H2g = -7.158 # eV
E_CO2g = -18.459 # eV
E_H2Og = -12.833 # eV
E_COg = -12.118 # eV

# G_gas = E_pot + E_ZPE + C_Idel_gas - TS + Ecorr_overbing + E_solvent
Gcor_H2g = 0.274 + 0.091 - 0.403 + 0.1 + 0 # eV
Gcor_CO2g = 0.306 + 0.099 - 0.664 + 0.3 + 0 # eV
Gcor_H2Og = 0.572 + 0.104 - 0.670 # eV
Gcor_COg = 0.132 + 0.091 - 0.669 # eV
G_H2g = E_H2g + Gcor_H2g
G_CO2g = E_CO2g + Gcor_CO2g
G_H2Og = E_H2Og + Gcor_H2Og
G_COg = E_COg + Gcor_COg

# G_gas = E_ZPE + C_harm - TS + Ecorr_overbing + E_solvent
Gcor_H = 0.190 + 0.003 - 0.004 + 0 + 0 # eV
Gcor_HOCO = 0.657 + 0.091 - 0.162 + 0.15 - 0.25 # eV
Gcor_CO = 0.186 + 0.080 - 0.156 + 0 - 0.10 # eV
Gcor_OH = 0.355 + 0.056 - 0.103 # eV

ddG_HOCO = 0.414
ddG_CO =  0.456
# ddG_CO = 0.579
"""
ddG_HOCO explanation:
    ddG_HOCO is the correction for the whole first equations, not for only *HOCO
    CO2(g) + * + 1/2 H2(g) -> HOCO*
    
    ddG_HOCO = Gcor_HOCO - Gcor_CO2g - 0 - 0.5 * Gcor_H2g 
    >> 0.41400000000000003

ddG_CO explanation:
    ddG_CO is the correction for the whole CO binding equations, not for the second equations
    * + CO(g) -> CO*
    
    ddG_CO = Gcor_CO - 0 - Gcor_COg
    >> 0.45600000000000007
"""

