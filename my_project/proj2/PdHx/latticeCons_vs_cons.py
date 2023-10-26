# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:40:15 2023

@author: changai
"""

import matplotlib.pyplot as plt
import pcat.utils.constants as const

def calc_binding_energy(ads='CO', E_Surface=0, E_CO=0, E_HOCO=0):
    E_H2g = const.E_H2g
    E_CO2g = const.E_CO2g
    E_H2Og = const.E_H2Og
    E_COg = const.E_COg
    E_Surface = E_Surface
    if ads == 'CO':
        E_CO = E_CO
        Eb_CO = E_CO - E_Surface - E_COg
        print(Eb_CO)
    elif ads == 'H':
        E_H = 0
        Eb_H = E_H - E_Surface - 0.5 * E_H2g
    elif ads == 'HOCO':
        E_HOCO = E_HOCO
        Eb_HOCO = E_HOCO - E_Surface - E_CO2g - 0.5 * E_H2g
        print(Eb_HOCO)
    return None

def latticeCons_vs_cons():
    tags = ['Pd', 'Pd64H39', 'Pd64H64']
    xs = [0, 0.608, 1]
    ys = [3.971, 4.065, 4.138]
    plt.figure(dpi=300)
    plt.plot(xs, ys, 'o')
    plt.plot([0, 1], [3.971, 4.138], 'r--')
    plt.xlabel('H concentration')
    plt.ylabel(f'Lattice constant ($\AA$)')
    plt.show()

if __name__ == '__main__':
    latticeCons_vs_cons()
    # ~/ce_PdHx/mc_round8/dft_candidates/18_Pd64X153H39/hollow1/CO
    calc_binding_energy(ads='CO', E_Surface=-247.928, E_CO=-260.80178969) # origin; fix bottom two layers; no compress
    
    
    # The following is after 0.017 compression of lattic constant
    # CO -> -0.756 # ~/ce_PdHx/diff_ads_site/run_dft_all_sites_candidates/adsorption/36_Pd64XH39/top1/CO/compress-0.017
    calc_binding_energy(ads='CO', E_Surface=-248.21332239, E_CO=-260.99111348) # fix all slab; CO* is the old; compress
    # calc_binding_energy(ads='CO', E_Surface=-247.90568847, E_CO=-261.14717784) # add CO* later; fix all slab; compress
    
    # HOCO* -> 0.272 eV
    calc_binding_energy(ads='HOCO', E_Surface=-248.21332239, E_CO=0, E_HOCO=-269.92647295) # fix all slab; HOCO* is the old; compress
    # calc_binding_energy(ads='HOCO', E_Surface=-247.90568847, E_CO=0, E_HOCO=-269.53102170) # add HOCO* later; fix all slab;  compress
    
    
    # fix slab without compression; add CO* later. ~/ce_PdHx/mc_round8/dft_candidates/18_Pd64X153H39/top1/surface
    calc_binding_energy(ads='CO', E_Surface=-247.928, E_CO=-260.93846732) # # start from dft structure; fix all slab; add CO*; no compression
    calc_binding_energy(ads='CO', E_Surface=-247.928, E_CO=-258.03) # # start from dft structure; fix bottom two layers; add CO*; no compression
    
    # calc_binding_energy(ads='CO', E_Surface=-243.25338753, E_CO=-256.67432772) # start from ce structure; fix all slab; add CO*; no compression
   