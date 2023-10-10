from ase.neighborlist import NeighborList
from ase.io import read
from ase.visualize import view
import pandas as pd

def check_distortion(atoms, cutoff=1.2):
    """Check if there are some distortion structures, such as, H2O, H2"""
    cutoff = cutoff # within 1.2 A
    nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms),
                            self_interaction=True,
                            bothways=True,
                            skin=0.)
    nl.update(atoms)
    for atom in atoms:
        if atom.symbol == 'O':
            n1 = atom.index
            indices, _ = nl.get_neighbors(n1)
            indices = list(set(indices))
            indices.remove(n1)
            if len(indices) != 0:
                syms = atoms[indices].get_chemical_symbols()
                num = syms.count('H')
                if num >= 2:
                    reason = 'H2O_exists'
                    return True, reason
        elif atom.symbol == 'H':
            n1 = atom.index
            indices, _ = nl.get_neighbors(n1)
            indices = list(set(indices))
            indices.remove(n1)
            if len(indices) != 0:
                syms = atoms[indices].get_chemical_symbols()
                num = syms.count('H')
                if num >= 1:
                    reason = 'H2_exists'
                    return True, reason
    return False, 'undistortion'

if __name__ == '__main__':
    images = read('vasp_PdTiH_adss_r1_final.traj', ':')
    names, distortions, reasons = [], [], []
    for row_id, atoms in enumerate(images):
        name = atoms.get_chemical_formula(mode='hill')
        job_id = str(row_id) + '_' + name
        distortion, reason = check_distortion(atoms)
        print(name, distortion, reason)
        names.append(name)
        distortions.append(distortion)
        reasons.append(reason)
    tuples = {
        'name': names,
        'distortion': distortions,
        'reason': reasons,
        }
    df = pd.DataFrame(tuples)
    df.to_csv('log_distortion.csv')
    
