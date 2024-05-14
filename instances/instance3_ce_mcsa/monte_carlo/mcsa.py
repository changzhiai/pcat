from ase import Atoms
from ase.build import surface, bulk,
from ase.visualize import view
import numpy as np
from ase.io import write,read
from ase.geometry import get_duplicate_atoms
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
from contextlib import contextmanager
from multiprocessing import Pool
from clease.settings import settings_from_json
from clease.calculator import attach_calculator
from clease.tools import species_chempot2eci
from clease.montecarlo import Montecarlo, RandomFlip, TrialMoveGenerator, RandomSwap
from clease.montecarlo.trial_move_generator import SingleTrialMoveGenerator
from clease.montecarlo.trial_move_generator import DualRamdomSwap
from clease.montecarlo.observers import ConcentrationObserver
from ase.db import connect
import json
import argparse
from clease.montecarlo.observers import Snapshot, LowestEnergyStructure
from copy import deepcopy
import logging
from ase.calculators.singlepoint import SinglePointCalculator
from clease.montecarlo.constraints import FixedElement
from ase.io.trajectory import TrajectoryWriter
from clease.tools import add_file_extension
from clease.montecarlo.observers import MCObserver
from clease.tools import reconfigure
from clease.calculator import get_ce_energy
from ase.calculators.singlepoint import SinglePointCalculator as SPC

logger = logging.getLogger(__name__)

num_H = sys.argv[1]
num_H = int(num_H)

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

def add_X(atoms):
    thickness = 2.389
    z_max_cell = atoms.cell[2, 2]
    top_metal_layer = atoms[[atom.index for atom in atoms if atom.z > 7 and atom.z < 8]]
    for atom in top_metal_layer:
        atom.symbol = 'X'
    while max(top_metal_layer.positions[:,2]) + thickness <  z_max_cell:
        top_metal_layer.positions[:,2] = top_metal_layer.positions[:,2] + thickness
        atoms+=top_metal_layer
    return atoms

@contextmanager
def cd(newdir):
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

def evaluate(num=0):
    from clease import Evaluate
    import clease.plot_post_process as pp
    eva = Evaluate(settings=settings, fitting_scheme='l1', scoring_scheme='k-fold', nsplits=10)
    alphas, cvs = eva.alpha_CV(alpha_min=1E-7, alpha_max=1.0, num_alpha=50)
    idx = cvs.argmin()
    alpha = alphas[idx]
    eva.set_fitting_scheme(fitting_scheme='l1', alpha=alpha)
    eva.get_eci()
    eva.save_eci(fname='eci_l1_{}.json'.format(str(num)))
    eci = eva.get_eci_dict()
    fig1 = pp.plot_cv(eva)
    fig1.savefig('cv_plot_{}.png'.format(str(num)))
    fig2 = pp.plot_fit(eva)
    fig2.savefig('eva_fitting_{}.png'.format(str(num)))
    fig3 = pp.plot_convex_hull(eva)
    fig3.savefig('convex_hull_{}.png'.format(str(num)))
    fig4 = pp.plot_eci(eva)
    fig4.savefig('plot_eci_{}.png'.format(str(num)))

class LowestEnergyStructure_snap(MCObserver):
    """Track the lowest energy state visited during an MC run.

    atoms: Atoms object
        Atoms object used in Monte Carlo

    verbose: bool
        If `True`, progress messages will be printed
    """
    name = "LowestEnergyStructure"

    def __init__(self, atoms, verbose=False):
        super().__init__()
        self.atoms = atoms
        self.lowest_energy = np.inf
        self.lowest_energy_cf = None
        self.emin_atoms = None
        self.verbose = verbose
        self.traj = TrajectoryWriter('gs_snap.traj', mode='w')

    def reset(self):
        self.lowest_energy = np.inf
        self.lowest_energy_cf = None
        self.emin_atoms = None

    @property
    def calc(self):
        return self.atoms.calc

    @property
    def energy(self):
        return self.calc.results['energy']

    def __call__(self, system_changes):
        """
        Check if the current state has lower energy and store the current
        state if it has a lower energy than the previous state.

        system_changes: list
            System changes. See doc-string of
            `clease.montecarlo.observers.MCObserver`
        """
        if self.emin_atoms is None or self.energy < self.lowest_energy:
            self._update_emin()
            dE = self.energy - self.lowest_energy
            self.traj.write(self.atoms)
            msg = "Found new low energy structure. "
            msg += f"New energy: {self.lowest_energy} eV. Change: {dE} eV"
            if self.verbose:
                print(msg)

    def _update_emin(self):
        self.lowest_energy_cf = self.calc.get_cf()
        self.lowest_energy = self.energy

        # Store emin atoms, and attach a cache calculator
        self.emin_atoms = self.atoms.copy()
        calc_cache = SinglePointCalculator(self.emin_atoms, energy=self.energy)
        self.emin_atoms.calc = calc_cache


if __name__ == '__main__':
    num = 0
    system = build_system()
    origin_with_H = system.repeat((2,2,1))
    system = add_X(system)

    import clease
    from clease.settings import ClusterExpansionSettings
    from clease.settings import Concentration
    from clease.template_filters import SlabCellFilter

    conc = Concentration(basis_elements=[['Pd', 'Ti'], ['H', 'X'],
                                         ['X']  # This is a "ghost" sublattice for vacuum
                                         ])
    conc.set_conc_formula_unit(formulas=['Pd<x>Ti<1-x>', 'H<y>X<1-y>', 'X'], variable_range={'x': (0., 1.),'y': (0, 1.)})

    prim = system
    concentration = conc
    size= [2, 2, 1]
    max_cluster_dia=[6.0, 5.0, 4.0]
    max_cluster_size=4
    supercell_factor=27

    db_name = 'clease.db'
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
    # reconfigure(settings)

    with open('../../4body/eci_l1_PdTiH.json') as f:
        eci = json.load(f)

    #set your atoms for mc
    origin_with_H = add_X(origin_with_H)
    list_H = [a.index for a in origin_with_H if a.symbol == 'H']
    ids_H = np.random.choice(list_H, size=num_H, replace=False)
    for index in ids_H:
        origin_with_H[index].symbol = 'X'


    list_Pd = [a.index for a in origin_with_H if a.symbol == 'Pd']
    def run(num_Ti):
        with cd('Ti{0}'.format(str(num_Ti))):
            atoms = deepcopy(origin_with_H)
            ids = np.random.choice(list_Pd, size=num_Ti, replace=False)
            print(ids)
            for i in ids:
                atoms[i].symbol = 'Ti'

            atoms = attach_calculator(settings, atoms, eci)
            # view(atoms)
            print(atoms)
            print(num_Ti, num_H)

            try:
                sites_Pd = [a.index for a in atoms if a.symbol == 'Pd']
            except:
                sites_Pd = []
            try:
                sites_Ti = [a.index for a in atoms if a.symbol == 'Ti']
            except:
                sites_Ti = []
            try:
                sites_H = [a.index for a in atoms if a.symbol == 'H']
            except:
                sites_H = []
            try:
                sites_X = [a.index for a in atoms if a.symbol == 'X' and a.z < 9]
            except:
                sites_X = []

            db_res = connect('result.db')
            sites_metals = sites_Pd + sites_Ti
            sites_HX = sites_H + sites_X
            if (sites_Pd == [] or sites_Ti == []) and (sites_H != [] and sites_X != []):
                generator =  RandomSwap(atoms, sites_HX)
            elif (sites_H == [] or sites_X == []) and (sites_Pd != [] and sites_Ti != []):
                generator =  RandomSwap(atoms, sites_metals)
            elif sites_Pd != [] and sites_Ti != [] and sites_H != [] and sites_X != []:
                generator = DualRamdomSwap(atoms, sites_metals, sites_HX)
            elif (sites_Pd == [] or sites_Ti == []) and (sites_H == [] or sites_X == []):
                clease_energy = get_ce_energy(settings, atoms, eci)
                # db_res.write(atoms, energy=clease_energy)
                new_calc = SPC(atoms, energy=clease_energy)
                atoms.set_calculator(new_calc) 
                db_res.write(atoms)
                return False

            mc = Montecarlo(atoms, 0, generator)

            conc_Pd = ConcentrationObserver(atoms, element='Pd')
            conc_Ti = ConcentrationObserver(atoms, element='Ti')
            conc_H = ConcentrationObserver(atoms, element='H')
            # conc_Ti = ConcentrationObserver(atoms, element='Ti')
            print('Pd and Ti cons:', conc_Pd, conc_Ti)
            sweeps=1000 
            temps = [1e10, 10000, 6000, 4000, 2000, 1500, 1000, 800, 700, 600, 500, 400, 350, 300, 250, 200, 150, 100, 75, 50, 25, 2, 1]
            # snap = Snapshot(fname=f'snapshot.traj', atoms=atoms, mode='w')
            les=LowestEnergyStructure_snap(atoms, verbose=True)
            runID = np.random.randint(0, 2**32-1)
            for T in temps:
                mc.attach(conc_Pd)
                mc.attach(conc_Ti)
                mc.attach(conc_H)
                mc.attach(les) # get lowest energy structures
                mc.T = T
                mc.run(steps=sweeps*len(atoms))
                thermo = mc.get_thermodynamic_quantities()
                a = mc.atoms
                db_res.write(a, data= thermo) 


    num_cores = os.cpu_count()
    with Pool(num_cores) as p:
        p.map(run, range(0, 65))
        p.close()

    print('finished')
