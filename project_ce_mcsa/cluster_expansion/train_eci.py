from ase import Atoms
from ase.build import surface
from ase.visualize import view
import numpy as np
from ase.io import write,read
from ase.geometry import get_duplicate_atoms
import os
from ase.constraints import FixAtoms
from clease.structgen import NewStructures
from ase.calculators.emt import EMT
from ase.db import connect
from clease.tools import update_db
import json
import sys
from clease.tools import reconfigure
import warnings
warnings.filterwarnings('ignore')

name = 'PdTiH'

def build_system(a = 4.138):
    bulk_PdH = Atoms(
        "Pd4H4",
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    )
    slab = surface(bulk_PdH, (1, 1, 1), 4)
    slab.center(vacuum=10, axis=2)    
    system = slab.copy()
    system.positions[:,2] = system.positions[:,2]-10.
    return system

def add_X(system):
    thickness = 2.389
    z_max_cell = system.cell[2, 2]
    top_metal_layer = system[[atom.index for atom in system if atom.z > 7 and atom.z < 8]]
    for atom in top_metal_layer:
        atom.symbol = 'X'
    while max(top_metal_layer.positions[:,2]) + thickness <  z_max_cell:
        top_metal_layer.positions[:,2] = top_metal_layer.positions[:,2] + thickness
        system+=top_metal_layer


if __name__ == '__main__':
    system = build_system()
    origin_with_H = system.repeat((2,2,1))
    add_X(system)

    import clease
    from clease.settings import ClusterExpansionSettings
    from clease.settings import Concentration
    from clease.template_filters import SlabCellFilter

    conc = Concentration(basis_elements=[['Pd', 'Ti'], ['H', 'X'], 
                                         ['X']  # This is a "ghost" sublattice for vacuum
                                         ])
    conc.set_conc_formula_unit(formulas=['Pd<x>Ti<1-x>', 'H<y>X<1-y>', 'X'], variable_range={'x': (0., 1.),'y': (0, 1.)})


    print('Starting setttings')
    prim = system
    concentration = conc
    size= [2, 2, 1]
    max_cluster_dia=[6.0, 5.0, 4.0]
    max_cluster_size=4
    supercell_factor=27
    db_name = '../PdTiH_surf_rn.db' # 200 slabs
    settings = ClusterExpansionSettings(prim,
                                        concentration,
                                        size=size,
                                        max_cluster_dia=max_cluster_dia,
                                        max_cluster_size=max_cluster_size,
                                        supercell_factor=supercell_factor,
                                        db_name=db_name,
                                        )
    cell_filter = SlabCellFilter(prim.cell)
    settings.template_atoms.add_cell_filter(cell_filter)
    settings.include_background_atoms = True
    # settings.basis_func_type='binary_linear' # using binary linear
    # reconfigure(settings)
    print('finished settings')

    from clease import Evaluate
    import clease.plot_post_process as pp
    eva = Evaluate(settings=settings, fitting_scheme='l1', scoring_scheme='k-fold', nsplits=10)
    # scan different values of alpha and return all values of alpha and its corresponding CV scores
    alphas, cvs = eva.alpha_CV(alpha_min=1E-7, alpha_max=1.0, num_alpha=50)
    # set the alpha value to the one that yields the lowest CV score, and fit data using it.
    idx = cvs.argmin()
    alpha = alphas[idx]
    eva.set_fitting_scheme(fitting_scheme='l1', alpha=alpha)
    eva.get_eci()
    # save a dictionary containing cluster names and their ECIs
    eva.save_eci(fname='eci_l1_{}.json'.format(str(name)))
    eci = eva.get_eci_dict()

    fig1 = pp.plot_cv(eva)
    fig1.savefig('cv_plot_{}.png'.format(str(name)))

    fig2 = pp.plot_fit(eva)
    fig2.savefig('eva_fitting_{}.png'.format(str(name)))

    # plot convex hull.
    fig3 = pp.plot_convex_hull(eva)
    fig3.savefig('convex_hull_{}.png'.format(str(name)))

    # # plot ECI values, an emptry list is passed to ignore_sizes to plot all cluster sizes 
    fig4 = pp.plot_eci(eva)
    fig4.savefig('plot_eci_{}.png'.format(str(name)))
