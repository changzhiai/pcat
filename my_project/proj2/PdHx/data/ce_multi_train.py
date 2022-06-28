# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:37:29 2022

@author: changai
"""

from ase import Atom, Atoms
from ase.visualize import view
import numpy as np
from ase.io import write, read
import os
from ase.constraints import FixAtoms
from clease.structgen import NewStructures
from ase.calculators.emt import EMT
from ase.db import connect
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
from clease.tools import update_db
import json
import sys
from ase.visualize import view
import clease
from clease.settings import ClusterExpansionSettings
from clease.settings import Concentration
from clease.settings.template_filters import SlabCellFilter
from contextlib import contextmanager
from multiprocessing import Pool
from clease.tools import reconfigure
from clease import Evaluate
import clease.plot_post_process as pp
from clease.calculator import get_ce_energy
from clease.calculator import attach_calculator
from clease.tools import wrap_and_sort_by_position
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.build import surface
import matplotlib as mpl
mpl.use('TkAgg')

# name = sys.argv[1]


@contextmanager
def cd(newdir):
    """Create directory and walk in"""
    prevdir = os.getcwd()
    try:
        os.makedirs(newdir)
    except OSError:
        pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def build_system():
    a = 4.138
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
    """Add customized vaccum layers, only for this system"""
    thickness = 2.389
    z_max_cell = system.cell[2, 2]
    top_metal_layer = system[
        [atom.index for atom in system if atom.z > 7 and atom.z < 8]
    ]
    for atom in top_metal_layer:
        atom.symbol = "X"
    while max(top_metal_layer.positions[:, 2]) + thickness < z_max_cell:
            top_metal_layer.positions[:, 2] = top_metal_layer.positions[:, 2] + thickness
            system += top_metal_layer
    return system

def rm_1X(atoms):
    """Remove the first X layer near to the surface"""
    del atoms[[atom.index for atom in atoms if atom.z > 9.555 and atom.z < 9.557]]
    return atoms


def top_pos(slab, i):
    """Get one top site position that can be used for adding adsorbates, only for this system"""
    top_site = slab[i].position
    return top_site


def hollow_pos(slab, i1, i2, i3):
    """Get one hollow site position that can be used for adding adsorbates, only for this system"""
    hollow_site = (
        1 / 3.0 * (slab[i1].x + slab[i2].x + slab[i3].x),
        1 / 3.0 * (slab[i1].y + slab[i2].y + slab[i3].y),
        1 / 3.0 * (slab[i1].z + slab[i2].z + slab[i3].z),
    )
    hollow_site = np.asarray(hollow_site)  # transfer type of variable to array
    return hollow_site


def add_ads(adsorbate, position, bg=False):
    """Add adsorbates on a specific site"""
    if adsorbate == "HOCO":
        ads = Atoms(
            [
                Atom("H", (0.649164000, -1.51784000, 0.929543000)),
                Atom("C", (0.000000000, 0.000000000, 0.000000000)),
                Atom("O", (-0.60412900, 1.093740000, 0.123684000)),
                Atom("O", (0.164889000, -0.69080700, 1.150570000)),
            ]
        )
        ads.translate(position + (0.0, 0.0, 3.0))
    elif adsorbate == "CO":
        ads = Atoms([Atom("C", (0.0, 0.0, 0.0)), Atom("O", (0.0, 0.0, 1.14))])
        ads.translate(position + (0.0, 0.0, 2.0))
    elif adsorbate == "H":
        ads = Atoms([Atom("H", (0.0, 0.0, 0.0))])
        ads.translate(position + (0.0, 0.0, 1.5))
    elif adsorbate == "OH":
        ads = Atoms([Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.97))])
        ads.translate(position + (0.0, 0.0, 2.0))
    if bg == True:
        for atom in ads:
            atom.symbol = "X"
    return ads

def run(args):
    print(args)
    return run_ce(*args)

def run_ce(site, adsor, insert=True, evaluate=True, predict=False, do_reconfigure=False):
    for adsorbate in [adsor]:
        # with cd("{0}_{1}_4body".format(site, adsorbate)):
        """Walk in this directory and some basic settings for slab with adsorbate"""
        site = site
        adsorbate = adsorbate
        name = site + "_" + adsorbate

        if site == "top1":
            index_num = 162

        print("Making template")
        # system = build_system()
        # system = system.repeat((2,2,1))
        # system = add_X(system)
        # remove_H(system)
        db_template = connect("/home/energy/changai/ce_PdxTiHy/random/mc/ce_ads/db/template.db")
        row1 = list(db_template.select(id=1))[0]
        system = row1.toatoms()
        # view(system)
        # import pdb; pdb.set_trace()

        # Add adsorbates layer (only one adsorbate, the rest of them are ghost atoms) on template slab
        for atoms in system:
            if atoms.z > 6.5 and atoms.z < 7.5:
                index = atoms.index
                pos = top_pos(system, index)
                if index != 162:
                    ads_bg = add_ads(adsorbate, pos, bg=True)
                    system.extend(ads_bg)
                else:
                    ads = add_ads(adsorbate, pos)
                    system.extend(ads)
        system = rm_1X(system)
        # view(system)
        # assert False
        mask = [atom.z < 4.0 for atom in system]
        system.set_constraint(FixAtoms(mask=mask))

        if adsorbate == "HOCO":
            conc = Concentration(
                basis_elements=[
                    ["Pd"],
                    ["H", "X"],
                    ["X"],  # This is a "ghost" sublattice for vacuum
                    ["C"],
                    ["O"],
                ]
            )
            conc.set_conc_formula_unit(
                formulas=["Pd", "H<x>X<1-x>", "X", "C", "O"],
                variable_range={"x": (0.0, 1.0)},
            )
        # *CO
        elif adsorbate == "CO":
            conc = Concentration(
                basis_elements=[
                    ["Pd"],
                    ["H", "X"],
                    ["X"],  # This is a "ghost" sublattice for vacuum
                    ["C"],
                    ["O"],
                ]
            )
            conc.set_conc_formula_unit(
                formulas=["Pd", "H<x>X<1-x>", "X", "C", "O"],
                variable_range={"x": (0.0, 1.0),},
            )
        # *H
        elif adsorbate == "H":
            conc = Concentration(
                basis_elements=[
                    ["Pd"],
                    ["H", "X"],
                    ["X"],  # This is a "ghost" sublattice for vacuum
                ]
            )
            conc.set_conc_formula_unit(
                formulas=["Pd", "H<x>X<1-x>", "X"],
                variable_range={"x": (0.0, 1.0),},
            )
        # *OH
        elif adsorbate == "OH":
            conc = Concentration(
                basis_elements=[
                    ["Pd"],
                    ["H", "X"],
                    ["X"],  # This is a "ghost" sublattice for vacuum
                    ["O"],
                ]
            )
            conc.set_conc_formula_unit(
                formulas=["Pd", "H<x>X<1-x>", "X", "O"],
                variable_range={"x": (0.0, 1.0)},
            )

        prim = system
        # prim = wrap_and_sort_by_position(system.copy())
        concentration = conc
        size = [1, 1, 1]
        max_cluster_dia = [6., 5., 4.]
        db_name = "../PdHx_top1_CO_r0.db"
        print("Making settings")
        settings = ClusterExpansionSettings(
            prim,
            concentration,
            size=size,
            max_cluster_dia=max_cluster_dia,
            db_name=db_name,
            include_background_atoms=True,
        )
        settings.save("settings.json")

        cell_filter = SlabCellFilter(prim.cell)
        # Ensure this cell filter persists through saving/loading with the jsonio module.
        settings.template_atoms.add_cell_filter(cell_filter)
        if do_reconfigure:
            print("Reconfiguring")
            reconfigure(settings, verbose=True)

        if insert == True:
            """Insert all initial structures (slab with vaccum layers and adsorbate layers) and dft calculated sturactures (slab with adsorbate calculated by VASP) into clease database"""
            print("Running insert")
            db_insert = connect("../PdHx_top1_CO_r1.db")

            new_struct = NewStructures(settings, check_db=False)
            for row in db_insert.select(struct_type="initial"):
                atoms_init = row.toatoms()
                uniqueid = row.get('formula_unit')
                uni_name = "PdH_CO_r1_" + str(row.id)
                print(uni_name)
                try:
                    new_struct.insert_structure(
                        init_struct=atoms_init, name=uni_name
                    )  # insert initial structures
                except ValueError:
                    # Already exists
                    continue
                
                final_atoms = db_insert.get_atoms(uniqueid=uniqueid, struct_type='final')
                db_clease = connect(db_name)
                uni_id = db_clease.get(name=uni_name).id
                # print(uni_id)
                update_db(
                    uid_initial=uni_id,
                    final_struct=final_atoms,
                    db_name=db_name,
                )  # insert final structures
                print('inserted: {}'.format(uni_id))

        if evaluate == True:
            """Evaluate and plot them"""
            print("Running evaluate")
            eva = Evaluate(
                settings=settings,
                fitting_scheme="l1",
                scoring_scheme="k-fold",
                nsplits=10,
            )
            # scan different values of alpha and return all values of alpha and its corresponding CV scores
            alphas, cvs = eva.alpha_CV(alpha_min=1e-7, alpha_max=1.0, num_alpha=50)
            # set the alpha value to the one that yields the lowest CV score, and fit data using it.
            idx = cvs.argmin()
            alpha = alphas[idx]
            eva.set_fitting_scheme(fitting_scheme="l1", alpha=alpha)
            eva.get_eci()
            # save a dictionary containing cluster names and their ECIs
            eva.save_eci(fname="eci_l1_{}.json".format(str(name)))
            eci = eva.get_eci_dict()

            fig1 = pp.plot_cv(eva)
            fig1.savefig("cv_plot_{}.png".format(str(name)))

            fig2 = pp.plot_fit(eva)
            fig2.savefig("eva_fitting_{}.png".format(str(name)))

            # plot convex hull.
            fig3 = pp.plot_convex_hull(eva)
            fig3.savefig("convex_hull_{}.png".format(str(name)))

            # # plot ECI values, an emptry list is passed to ignore_sizes to plot all cluster sizes
            fig4 = pp.plot_eci(eva)
            fig4.savefig("plot_eci_{}.png".format(str(name)))

        if predict == True:
            """Predict new structures"""
            print("Running predict")
            with open("./eci_l2_{}_{}.json".format(site, adsorbate)) as f:
                eci = json.load(f)

            db_predict = connect("/home/energy/changai/ce_PdxTiHy/random/mc/ce_ads/db/candidates.db")
            db_result = connect(
                "collect_predict_candidates_{}_{}.db".format(site, adsorbate)
            )
    
            for row in db_predict.select():
                atoms = row.toatoms()

                # Add adsorbates layer (only one adsorbate, the rest of them are ghost atoms) on each slab
                for atom in atoms:
                    if atom.z > 6.5 and atom.z < 7.5:
                        index = atom.index
                        pos = top_pos(atoms, index)
                        if index != index_num:
                            ads_bg = add_ads(
                                adsorbate, pos, bg=True
                            )  # add adsorbate
                            atoms.extend(ads_bg)
                        else:
                            ads_bg = add_ads(adsorbate, pos)
                            atoms.extend(ads_bg)

                atoms = rm_1X(atoms)
                print(atoms)
                atoms.calc = None
                a = attach_calculator(settings, atoms=atoms, eci=eci)
                clease_energy = get_ce_energy(settings, a, eci)
                calc = SPC(atoms=a, energy=clease_energy)
                a.set_calculator(calc)
                print(clease_energy)
                db_result.write(a)


if __name__ == "__main__":
    print("Running train")
    # train
    # sites = ['top1']
    # # sites = ['top1', 'hollow1']
    # adsors = ['HOCO', 'CO', 'H', 'OH']
    
    # args = []
    # for site in sites:
    #     for adsor in adsors:
    #         args.append((site, adsor))
    
    # with Pool(4) as p:
    #     p.map(run, args)
    #     p.close()
    #     p.join()  
    
    run_ce('top1', 'CO', insert=True, evaluate=False, predict=False, do_reconfigure=True)
    
    # print("Running predict")
    # predict
    # run("HOCO", insert=False, evaluate=True, predict=True, do_reconfigure=True)
