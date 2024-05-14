"""Operators that work on slabs and adsorbates.
Allowed compositions are respected.
Identical indexing of the slabs are assumed for the cut-splice operator."""
from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np

from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell

try:
    import spglib
except ImportError:
    spglib = None

from ase import Atom, Atoms
from ase.geometry import find_mic
import random
import copy
from scipy.spatial import Delaunay
from ase.neighborlist import NeighborList
from ase.visualize import view

def permute2(atoms, rng=np.random, element_pools=None):
    if not bool(element_pools):
        i1 = rng.choice(range(len(atoms)))
        sym1 = atoms[i1].symbol
        i2 = rng.choice([a.index for a in atoms if a.symbol != sym1])
        atoms[i1].symbol = atoms[i2].symbol
        atoms[i2].symbol = sym1
    else:
        list1 = [a.index for a in atoms if a.symbol == element_pools[0] and a.tag >= 0 and a.tag != 4] # non ads for tag setting
        list2 = [a.index for a in atoms if a.symbol == element_pools[1] and a.tag >= 0 and a.tag != 4]
        if bool(list1) and bool(list2):
            i1 = rng.choice(list1)
            sym1 = atoms[i1].symbol
            i2 = rng.choice(list2)
            atoms[i1].symbol = atoms[i2].symbol
            atoms[i2].symbol = sym1
        else:
            print('permute2 failed')

def replace_multiple(atoms, rng=np.random, element_pools=None, first_nums_list=[]):
    """first_atoms_nums is a list including allowed the number of atoms
    for example: first_nums_list = [1, 2, 3]"""
    if not bool(element_pools):
        print('please specify element pools')
        assert False
    elif first_nums_list != []:
        list1 = [a.index for a in atoms if a.symbol == element_pools[0] and a.tag >= 0 and a.tag != 4] # non ads for tag setting
        list2 = [a.index for a in atoms if a.symbol == element_pools[1] and a.tag >= 0 and a.tag != 4]
        if len(list1)-1 in first_nums_list and len(list1)+1 in first_nums_list:
            operate = rng.choice(['add', 'remove'])
            if operate == 'remove':
                i1 = rng.choice(list1) 
                atoms[i1].symbol = element_pools[1]
            elif operate == 'add':
                i2 = rng.choice(list2)
                atoms[i2].symbol = element_pools[0]
        elif len(list1)-1 in first_nums_list:
            i1 = rng.choice(list1) 
            atoms[i1].symbol = element_pools[1]
        elif len(list1)+1 in first_nums_list:
            i2 = rng.choice(list2)
            atoms[i2].symbol = element_pools[0]
        elif len(list1) > max(first_nums_list):
            random.seed(random.randint(1,100000000))
            random_nums = random.choice(first_nums_list)
            new_list1 = random.sample(list1, random_nums)
            for ls in [ls for ls in list1 if ls not in new_list1]:
                atoms[ls].symbol = element_pools[1]
        elif len(list1) < min(first_nums_list):
            random.seed(random.randint(1,100000000))
            random_nums = random.choice(first_nums_list)
            new_list1 = random.sample(list2, random_nums-len(list1))
            for ls in new_list1:
                atoms[ls].symbol = element_pools[0]
        else:
            print('replace error')
            assert False
    return atoms

def replace(atoms, rng=np.random, element_pools=None):
    """Replace one element to another existed element"""
    if not bool(element_pools):
        print('please specify element pools')
        assert False
    else:
        list1 = [a.index for a in atoms if a.symbol == element_pools[0] and a.tag >= 0 and a.tag != 4] # non ads for tag setting
        list2 = [a.index for a in atoms if a.symbol == element_pools[1] and a.tag >= 0 and a.tag != 4]
        if bool(list1) and bool(list2):
            operate = rng.choice(['add', 'remove'])
            if operate == 'remove':
                i1 = rng.choice(list1) 
                atoms[i1].symbol = element_pools[1]
            elif operate == 'add':
                i2 = rng.choice(list2)
                atoms[i2].symbol = element_pools[0]
        if bool(list1):
            i1 = rng.choice(list1) 
            atoms[i1].symbol = element_pools[1]
        elif bool(list2):
            i2 = rng.choice(list2)
            atoms[i2].symbol = element_pools[0]
        else:
            print('permute2 failed')
    return atoms

def replace_element(atoms, element_out, element_in):
    syms = np.array(atoms.get_chemical_symbols())
    syms[syms == element_out] = element_in
    atoms.set_chemical_symbols(syms)


def get_add_remove_lists(**kwargs):
    to_add, to_rem = [], []
    for s, amount in kwargs.items():
        if amount > 0:
            to_add.extend([s] * amount)
        elif amount < 0:
            to_rem.extend([s] * abs(amount))
    return to_add, to_rem


def get_minority_element(atoms):
    counter = Counter(atoms.get_chemical_symbols())
    return sorted(counter.items(), key=itemgetter(1), reverse=False)[0][0]


def minority_element_segregate(atoms, layer_tag=1, rng=np.random):
    """Move the minority alloy element to the layer specified by the layer_tag,
    Atoms object should contain atoms with the corresponding tag."""
    sym = get_minority_element(atoms)
    layer_indices = set([a.index for a in atoms if a.tag == layer_tag])
    minority_indices = set([a.index for a in atoms if a.symbol == sym])
    change_indices = minority_indices - layer_indices
    in_layer_not_sym = list(layer_indices - minority_indices)
    rng.shuffle(in_layer_not_sym)
    if len(change_indices) > 0:
        for i, ai in zip(change_indices, in_layer_not_sym):
            atoms[i].symbol = atoms[ai].symbol
            atoms[ai].symbol = sym


def same_layer_comp(atoms, rng=np.random):
    unique_syms, comp = np.unique(sorted(atoms.get_chemical_symbols()),
                                  return_counts=True)
    l = get_layer_comps(atoms)
    sym_dict = dict((s, int(np.array(c) / len(l)))
                    for s, c in zip(unique_syms, comp))
    for la in l:
        correct_by = sym_dict.copy()
        lcomp = dict(
            zip(*np.unique([atoms[i].symbol for i in la], return_counts=True)))
        for s, num in lcomp.items():
            correct_by[s] -= num
        to_add, to_rem = get_add_remove_lists(**correct_by)
        for add, rem in zip(to_add, to_rem):
            ai = rng.choice([i for i in la if atoms[i].symbol == rem])
            atoms[ai].symbol = add


def get_layer_comps(atoms, eps=1e-2):
    lc = []
    old_z = np.inf
    for z, ind in sorted([(a.z, a.index) for a in atoms]):
        if abs(old_z - z) < eps:
            lc[-1].append(ind)
        else:
            lc.append([ind])
        old_z = z

    return lc


def get_ordered_composition(syms, pools=None):
    if pools is None:
        pool_index = dict((sym, 0) for sym in set(syms))
    else:
        pool_index = {}
        for i, pool in enumerate(pools):
            if isinstance(pool, str):
                pool_index[pool] = i
            else:
                for sym in set(syms):
                    if sym in pool:
                        pool_index[sym] = i
    syms = [(sym, pool_index[sym], c)
            for sym, c in zip(*np.unique(syms, return_counts=True))]
    unique_syms, pn, comp = zip(
        *sorted(syms, key=lambda k: (k[1] - k[2], k[0])))
    return (unique_syms, pn, comp)


def dummy_func(*args):
    return


class SlabOperator(OffspringCreator):
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 distribution_correction_function=None,
                 element_pools=None,
                 rng=np.random):
        OffspringCreator.__init__(self, verbose, num_muts=num_muts, rng=rng)

        self.allowed_compositions = allowed_compositions
        self.element_pools = element_pools
        if distribution_correction_function is None:
            self.dcf = dummy_func
        else:
            self.dcf = distribution_correction_function
        # Number of different elements i.e. [2, 1] if len(element_pools) == 2
        # then 2 different elements in pool 1 is allowed but only 1 from pool 2

    def get_symbols_to_use(self, syms):
        """Get the symbols to use for the offspring candidate. The returned
        list of symbols will respect self.allowed_compositions"""
        if self.allowed_compositions is None:
            return syms

        unique_syms, counts = np.unique(syms, return_counts=True)
        comp, unique_syms = zip(*sorted(zip(counts, unique_syms),
                                        reverse=True))

        for cc in self.allowed_compositions:
            comp += (0,) * (len(cc) - len(comp))
            if comp == tuple(sorted(cc)):
                return syms

        comp_diff = self.get_closest_composition_diff(comp)
        to_add, to_rem = get_add_remove_lists(
            **dict(zip(unique_syms, comp_diff)))
        for add, rem in zip(to_add, to_rem):
            tbc = [i for i in range(len(syms)) if syms[i] == rem]
            ai = self.rng.choice(tbc)
            syms[ai] = add
        return syms

    def get_add_remove_elements(self, syms):
        if self.element_pools is None or self.allowed_compositions is None:
            return [], []
        unique_syms, pool_number, comp = get_ordered_composition(
            syms, self.element_pools)
        stay_comp, stay_syms = [], []
        add_rem = {}
        per_pool = len(self.allowed_compositions[0]) / len(self.element_pools)
        pool_count = np.zeros(len(self.element_pools), dtype=int)
        for pn, num, sym in zip(pool_number, comp, unique_syms):
            pool_count[pn] += 1
            if pool_count[pn] <= per_pool:
                stay_comp.append(num)
                stay_syms.append(sym)
            else:
                add_rem[sym] = -num
        # collect elements from individual pools
        diff = self.get_closest_composition_diff(stay_comp)
        add_rem.update(dict((s, c) for s, c in zip(stay_syms, diff)))
        return get_add_remove_lists(**add_rem)

    def get_closest_composition_diff(self, c):
        comp = np.array(c)
        mindiff = 1e10
        allowed_list = list(self.allowed_compositions)
        self.rng.shuffle(allowed_list)
        for ac in allowed_list:
            diff = self.get_composition_diff(comp, ac)
            numdiff = sum([abs(i) for i in diff])
            if numdiff < mindiff:
                mindiff = numdiff
                ccdiff = diff
        return ccdiff

    def get_composition_diff(self, c1, c2):
        difflen = len(c1) - len(c2)
        if difflen > 0:
            c2 += (0,) * difflen
        return np.array(c2) - c1

    def get_possible_mutations(self, a):
        unique_syms, comp = np.unique(sorted(a.get_chemical_symbols()),
                                      return_counts=True)
        min_num = min([i for i in np.ravel(list(self.allowed_compositions))
                       if i > 0])
        muts = set()
        for i, n in enumerate(comp):
            if n != 0:
                muts.add((unique_syms[i], n))
            if n % min_num >= 0:
                for j in range(1, n // min_num):
                    muts.add((unique_syms[i], min_num * j))
        return list(muts)

    def get_all_element_mutations(self, a):
        """Get all possible mutations for the supplied atoms object given
        the element pools."""
        muts = []
        symset = set(a.get_chemical_symbols())
        for sym in symset:
            for pool in self.element_pools:
                if sym in pool:
                    muts.extend([(sym, s) for s in pool if s not in symset])
        return muts

    def finalize_individual(self, indi):
        atoms_string = ''.join(indi.get_chemical_symbols())
        # indi.info['key_value_pairs']['atoms_string'] = atoms_string
        indi.info['data']['atoms_string'] = atoms_string
        return OffspringCreator.finalize_individual(self, indi)


class CutSpliceSlabCrossover(SlabOperator):
    def __init__(self, allowed_compositions=None, element_pools=None,
                 verbose=False,
                 num_muts=1, tries=1000, min_ratio=0.25,
                 distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              distribution_correction_function,
                              element_pools=element_pools,
                              rng=rng)

        self.tries = tries
        self.min_ratio = min_ratio
        self.descriptor = 'CutSpliceSlabCrossover'

    def get_new_individual(self, parents):
        f, m = parents

        indi = self.initialize_individual(f, self.operate(f, m))
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        parent_message = ': Parents {0} {1}'.format(f.info['confid'],
                                                    m.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, f, m):
        child = f.copy()
        fp = f.positions
        ma = np.max(fp.transpose(), axis=1)
        mi = np.min(fp.transpose(), axis=1)

        for _ in range(self.tries):
            # Find center point of cut
            rv = [self.rng.rand() for _ in range(3)]  # random vector
            midpoint = (ma - mi) * rv + mi

            # Determine cut plane
            theta = self.rng.rand() * 2 * np.pi  # 0,2pi
            phi = self.rng.rand() * np.pi  # 0,pi
            e = np.array((np.sin(phi) * np.cos(theta),
                          np.sin(theta) * np.sin(phi),
                          np.cos(phi)))

            # Cut structures
            d2fp = np.dot(fp - midpoint, e)
            fpart = d2fp > 0
            ratio = float(np.count_nonzero(fpart)) / len(f)
            if ratio < self.min_ratio or ratio > 1 - self.min_ratio:
                continue
            syms = np.where(fpart, f.get_chemical_symbols(),
                            m.get_chemical_symbols())
            dists2plane = abs(d2fp)

            # Correct the composition
            # What if only one element pool is represented in the offspring
            to_add, to_rem = self.get_add_remove_elements(syms)

            # Change elements closest to the cut plane
            for add, rem in zip(to_add, to_rem):
                tbc = [(dists2plane[i], i)
                       for i in range(len(syms)) if syms[i] == rem]
                ai = sorted(tbc)[0][1]
                syms[ai] = add

            child.set_chemical_symbols(syms)
            break

        self.dcf(child)

        return child


# Mutations: Random, MoveUp/Down/Left/Right, six or all elements
class RandomCompositionMutation(SlabOperator):
    """Change the current composition to another of the allowed compositions.
    The allowed compositions should be input in the same order as the element pools,
    for example:
    element_pools = [['Au', 'Cu'], ['In', 'Bi']]
    allowed_compositions = [(6, 2), (5, 3)]
    means that there can be 5 or 6 Au and Cu, and 2 or 3 In and Bi.
    """

    def __init__(self, verbose=False, num_muts=1, element_pools=None,
                 allowed_compositions=None,
                 distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              distribution_correction_function,
                              element_pools=element_pools,
                              rng=rng)

        self.descriptor = 'RandomCompositionMutation'

    def get_new_individual(self, parents):
        f = parents[0]
        parent_message = ': Parent {0}'.format(f.info['confid'])

        if self.allowed_compositions is None:
            if len(set(f.get_chemical_symbols())) == 1:
                if self.element_pools is None:
                    # We cannot find another composition without knowledge of
                    # other allowed elements or compositions
                    return None, self.descriptor + parent_message

        # Do the operation
        indi = self.initialize_individual(f, self.operate(f))
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        allowed_comps = self.allowed_compositions
        if allowed_comps is None:
            n_elems = len(set(atoms.get_chemical_symbols()))
            n_atoms = len(atoms)
            allowed_comps = [c for c in permutations(range(1, n_atoms),
                                                     n_elems)
                             if sum(c) == n_atoms]

        # Sorting the composition to have the same order as in element_pools
        syms = atoms.get_chemical_symbols()
        unique_syms, _, comp = get_ordered_composition(syms,
                                                       self.element_pools)

        # Choose the composition to change to
        for i, allowed in enumerate(allowed_comps):
            if comp == tuple(allowed):
                allowed_comps = np.delete(allowed_comps, i, axis=0)
                break
        chosen = self.rng.randint(len(allowed_comps))
        comp_diff = self.get_composition_diff(comp, allowed_comps[chosen])

        # Get difference from current composition
        to_add, to_rem = get_add_remove_lists(
            **dict(zip(unique_syms, comp_diff)))

        # Correct current composition
        syms = atoms.get_chemical_symbols()
        for add, rem in zip(to_add, to_rem):
            tbc = [i for i in range(len(syms)) if syms[i] == rem]
            ai = self.rng.choice(tbc)
            syms[ai] = add

        atoms.set_chemical_symbols(syms)
        self.dcf(atoms)
        return atoms


class RandomElementMutation(SlabOperator):
    def __init__(self, element_pools, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              distribution_correction_function,
                              element_pools=element_pools,
                              rng=rng)

        self.descriptor = 'RandomElementMutation'

    def get_new_individual(self, parents):
        f = parents[0]

        # Do the operation
        indi = self.initialize_individual(f, self.operate(f))
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        poss_muts = self.get_all_element_mutations(atoms)
        chosen = self.rng.randint(len(poss_muts))
        replace_element(atoms, *poss_muts[chosen])
        self.dcf(atoms)
        return atoms


class NeighborhoodElementMutation(SlabOperator):
    def __init__(self, element_pools, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              distribution_correction_function,
                              element_pools=element_pools,
                              rng=rng)

        self.descriptor = 'NeighborhoodElementMutation'

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.initialize_individual(f, f)
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        indi = self.operate(indi)

        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        least_diff = 1e22
        for mut in self.get_all_element_mutations(atoms):
            dist = get_periodic_table_distance(*mut)
            if dist < least_diff:
                poss_muts = [mut]
                least_diff = dist
            elif dist == least_diff:
                poss_muts.append(mut)

        chosen = self.rng.randint(len(poss_muts))
        replace_element(atoms, *poss_muts[chosen])
        self.dcf(atoms)
        return atoms


class SymmetrySlabPermutation(SlabOperator):
    """Permutes the atoms in the slab until it has a higher symmetry number."""

    def __init__(self, verbose=False, num_muts=1, sym_goal=10, max_tries=10,
                 allowed_compositions=None,
                 element_pools=None, # add element pools:)
                 distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              distribution_correction_function,
                              element_pools=element_pools, # :)
                              rng=rng)
        if spglib is None:
            print("SymmetrySlabPermutation needs spglib to function")

        assert sym_goal >= 1
        self.sym_goal = sym_goal
        self.max_tries = max_tries
        self.descriptor = 'SymmetrySlabPermutation'
        self.element_pools = element_pools

    def get_new_individual(self, parents):
        f = parents[0]
        flag = 0
        if bool(self.element_pools):
            list1 = [a.index for a in f if a.symbol == self.element_pools[0] and a.tag >= 0 and a.tag != 4]
            list2 = [a.index for a in f if a.symbol == self.element_pools[1] and a.tag >= 0 and a.tag != 4]
            if bool(list1) or bool(list2):
               flag = 1
        # Permutation only makes sense if two different elements are present
        if len(set(f.get_chemical_symbols())) == 1 or flag == 1:
            f = parents[1]
            if len(set(f.get_chemical_symbols())) == 1:
                return None, '{1} not possible in {0}'.format(f.info['confid'],
                                                              self.descriptor)

        indi = self.initialize_individual(f, self.operate(f))
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        sym_num = 1
        sg = self.sym_goal
        while sym_num < sg:
            for _ in range(self.max_tries):
                for _ in range(2):
                    permute2(atoms, rng=self.rng, element_pools=self.element_pools)
                self.dcf(atoms)
                sym_num = spglib.get_symmetry_dataset(
                    atoms_to_spglib_cell(atoms))['number']
                if sym_num >= sg:
                    break
            sg -= 1
        return atoms


class RandomMetalPermutation(SlabOperator):
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 element_pools=None, # add element pools:)
                 distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              distribution_correction_function,
                              element_pools=element_pools, # :)
                              rng=rng)

        self.descriptor = 'RandomSlabPermutation'
        self.element_pools = element_pools

    def get_new_individual(self, parents):
        f = parents[0]
        flag = 0
        if bool(self.element_pools):
            list1 = [a.index for a in f if a.symbol == self.element_pools[0] and a.tag >= 0 and a.tag != 4] # non ads for tag setting
            list2 = [a.index for a in f if a.symbol == self.element_pools[1] and a.tag >= 0 and a.tag != 4]
            if not bool(list1) or not bool(list2):
               flag = 1  
       # Permutation only makes sense if two different elements are present
        if len(set(f.get_chemical_symbols())) <= 2 or flag == 1:
            f = parents[1]
            if len(set(f.get_chemical_symbols())) <= 2:
                return None, '{1} not possible in {0}'.format(f.info['confid'],
                                                              self.descriptor)
                  
        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        indi = self.operate(indi)

        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        # print('before permutation:', atoms)
        for _ in range(self.num_muts):
            permute2(atoms, rng=self.rng, element_pools=self.element_pools)
            # permute2(atoms, rng=self.rng)
        self.dcf(atoms)
        # print('after permutation:', atoms, '\n')
        return atoms
    
class RandomMetalComposition(SlabOperator):
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 element_pools=None, # add element pools:)
                 distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              distribution_correction_function,
                              element_pools=element_pools, # :)
                              rng=rng)

        self.descriptor = 'RandomSlabComposition'
        self.element_pools = element_pools
        self.allowed_compositions = allowed_compositions
        """self.allowed_compositions is a list only including min concentration and max concentration
        for example allowed_compositions = [0.25, 0.75]"""

    def get_new_individual(self, parents):
        f = parents[0]
        flag = 0
        if bool(self.element_pools):
            list1 = [a.index for a in f if a.symbol == self.element_pools[0] and a.tag >= 0 and a.tag != 4] # non ads for tag setting
            list2 = [a.index for a in f if a.symbol == self.element_pools[1] and a.tag >= 0 and a.tag != 4]
            if not bool(list1) or not bool(list2):
               flag = 1  
       # Permutation only makes sense if two different elements are present
        if len(set(f.get_chemical_symbols())) <= 2 or flag == 1:
            f = parents[1]
            if len(set(f.get_chemical_symbols())) <= 2:
                return None, '{1} not possible in {0}'.format(f.info['confid'],
                                                              self.descriptor)
                  
        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]

        indi = self.operate(indi)

        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        for _ in range(self.num_muts):
            if self.allowed_compositions == None:
                atoms = replace(atoms, self.rng, self.element_pools)
            else:
                min_conc, max_conc = min(self.allowed_compositions), max(self.allowed_compositions)
                all_metals_indices = [atom.index for atom in atoms if atom.symbol in self.element_pools]
                assert len(all_metals_indices) > 0
                first_nums_list=[]
                for i in range(len(all_metals_indices)):
                    conc = float(i)/len(all_metals_indices)
                    if conc >= float(min_conc) and conc <= float(max_conc):
                        first_nums_list.append(i)
                atoms = replace_multiple(atoms, rng=np.random, element_pools=None, first_nums_list=first_nums_list)
        return atoms

class AdsorbateOperator(OffspringCreator):
    """Adsorbate operator"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 ads_pools=['CO', 'OH', 'H'],
                 rng=np.random, 
                 tc_cutoff=1.6): # too close cutoff
        OffspringCreator.__init__(self, verbose, num_muts=num_muts, rng=rng)

        self.allowed_compositions = allowed_compositions
        self.ads_pools = ads_pools
        self.tc_cutoff = tc_cutoff
        
    def atoms_too_close_after_addition(self, atoms, n_added, cutoff, mic=True): 
        """Check if there are atoms that are too close to each other after 
        adding some new atoms.

        n_added : int
            Number of newly added atoms.
        cutoff : float, default 1.5
            The cutoff radius. Two atoms are too close if the distance between
            them is less than this cutoff
        mic : bool, default False
            Whether to apply minimum image convention. Remember to set 
            mic=True for periodic systems.
        """
        newp, oldp = atoms.positions[-n_added:], atoms.positions[:-n_added]
        newps = np.repeat(newp, len(oldp), axis=0)
        oldps = np.tile(oldp, (n_added, 1))
        if mic:
            _, dists = find_mic(newps - oldps, atoms.cell, pbc=True)
        else:
            dists = np.linalg.norm(newps - oldps, axis=1)
        return any(dists < cutoff)
            
    def add_ads(self, site_pos, ads_symbol, bg=False):
        """Add adsorbates on a specific site"""
        if ads_symbol == "HOCO":
            ads = Atoms([
                    Atom("H", (0.649164000, -1.51784000, 0.929543000), tag=-4),
                    Atom("C", (0.000000000, 0.000000000, 0.000000000), tag=-4),
                    Atom("O", (-0.60412900, 1.093740000, 0.123684000), tag=-4),
                    Atom("O", (0.164889000, -0.69080700, 1.150570000), tag=-4),])
            ads.translate(site_pos + (0.0, 0.0, 2.5))
        elif ads_symbol == "CO":
            ads = Atoms([Atom("C", (0.0, 0.0, 0.0), tag=-2), 
                          Atom("O", (0.0, 0.0, 1.14), tag=-2)])
            ads.translate(site_pos + (0.0, 0.0, 2.0))
        elif ads_symbol == "H":
            ads = Atoms([Atom("H", (0.0, 0.0, 0.0), tag=-1)])
            ads.translate(site_pos + (0.0, 0.0, 2.0))
        elif ads_symbol == "OH":
            ads = Atoms([Atom("O", (0.0, 0.0, 0.0), tag=-3), 
                          Atom("H", (0.0, 0.0, 0.97), tag=-3)])
            ads.translate(site_pos + (0.0, 0.0, 2.0))
        elif bg == True:
            for atom in ads:
                atom.symbol = "X"
        else:
            assert False
        return ads
    
    def get_ads_pos(self, atoms, ads_index, ads_symbol):
        if ads_symbol == "HOCO":
            assert len(ads_index)==4
            site_pos = [atom.position for atom in atoms[ads_index] if atom.symbol=='C']
            assert len(site_pos) == 1
            site_pos = site_pos[0]
            site_pos = site_pos - (0.0, 0.0, 2.5) # reset
        elif ads_symbol == "CO":
            assert len(ads_index)==2
            site_pos = [atom.position for atom in atoms[ads_index] if atom.symbol=='C']
            assert len(site_pos) == 1
            site_pos = site_pos[0]
            site_pos = site_pos - (0.0, 0.0, 2.0)
        elif ads_symbol == "H":
            site_pos = [atom.position for atom in atoms[ads_index] if atom.symbol=='H']
            assert len(site_pos) == 1
            site_pos = site_pos[0]
            site_pos = site_pos - (0.0, 0.0, 2.0)
        elif ads_symbol == "OH":
            assert len(ads_index)==2
            site_pos = [atom.position for atom in atoms[ads_index] if atom.symbol=='O']
            assert len(site_pos) == 1
            site_pos = site_pos[0]
            site_pos = site_pos - (0.0, 0.0, 2.0) 
        else:
            raise ValueError('The adsorbate is currently not supported')
        return site_pos
    
    def debug_atoms_data(self, atoms, ads_indices, ads_symbols):
        print('===debug start===')
        print(ads_indices, ads_symbols)
        for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
            print(str(atoms[ads_index].symbols), ads_symbol)
        print('---ads location---')
        for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
            print(str(atoms[ads_index].symbols), ads_symbol)
            for i in ads_index:
                s = atoms[i].symbol
                assert (s in ads_symbol)
        print('===debug end===')
        return True
    
    def get_adsorbates_from_slab(self, atoms, debug=False):
        """Get adsorbate information from atoms, including indices and symbols"""
        ads_indices = atoms.info['data']['ads_indices']
        ads_symbols = atoms.info['data']['ads_symbols']
        assert len(ads_indices)==len(ads_symbols)
        if debug:
            self.debug_atoms_data(atoms, ads_indices, ads_symbols)
        return ads_indices, ads_symbols
    
    def update_all_adss_indices(self, atoms, atoms_old):
        """Update adsorbates indices after deleting one adsorbate"""
        update_ads_indices = []
        update_ads_symbols = []
        ads_indices_old, ads_symbols_old = self.get_adsorbates_from_slab(atoms_old, debug=False)
        if len(atoms) < len(atoms_old): # due to deleting adsorbate
            for ads_index_old, ads_symbol_old in zip(ads_indices_old, ads_symbols_old):
                update_ads_index = []
                for atom_index_old in ads_index_old:
                    atom_pos_old = atoms_old[atom_index_old].position
                    atoms_sym_old = atoms_old[atom_index_old].symbol
                    ads_indices_new = [atom.index for atom in atoms if atom.tag<0]
                    for atom_index_new in ads_indices_new:
                        atom_pos_new = atoms[atom_index_new].position
                        atoms_sym_new = atoms[atom_index_new].symbol
                        if (abs(atom_pos_new-atom_pos_old)<0.0000001).all() and atoms_sym_new==atoms_sym_old:
                            update_ads_index.append(atom_index_new)
                if update_ads_index != [] and len(update_ads_index)==len(ads_index_old):
                    update_ads_indices.append(update_ads_index)
                    update_ads_symbols.append(ads_symbol_old)
        return update_ads_indices, update_ads_symbols
    
    def remove_adsorbate_from_slab(self, atoms, ads_index, ads_symbol):
        """Remove one adsorbate"""
        atoms = copy.deepcopy(atoms)
        atoms_old = copy.deepcopy(atoms)
        del atoms[ads_index]
        print('deleting:', ads_index, ads_symbol)
        update_ads_indices, update_ads_symbols = self.update_all_adss_indices(atoms, atoms_old)
        atoms.info['data']['ads_symbols'] = update_ads_symbols
        atoms.info['data']['ads_indices'] = update_ads_indices
        _, _ = self.get_adsorbates_from_slab(atoms, debug=False)
        return atoms
    
    def add_adsorbate_onto_slab(self, atoms, site_pos, ads_symbol, cutoff):
        atoms = copy.deepcopy(atoms)
        atoms_old = copy.deepcopy(atoms)
        ads = self.add_ads(site_pos, ads_symbol, bg=False)
        atoms.extend(ads) 
        indices = [atom.index for atom in atoms]
        ads_index = indices[-len(ads):]
        print('adding:', ads_index, ads_symbol)
        if_too_close = self.atoms_too_close_after_addition(atoms, len(ads), cutoff, mic=True)
        if if_too_close:
            atoms = atoms_old
            print('too close, adding adsorbate failed')
        else:
            assert len(ads_index) == len(ads_symbol)
            atoms.info['data']['ads_indices'].append(ads_index)
            atoms.info['data']['ads_symbols'].append(ads_symbol)
        return atoms, if_too_close
        
    def substitute_adsorbate_on_slab(self, atoms, ads_index, ads_symbol):
        """Random substitue one adsorbate using other one on the same slab"""
        atoms = copy.deepcopy(atoms)
        _, _ = self.get_adsorbates_from_slab(atoms)
        site_pos = self.get_ads_pos(atoms, ads_index, ads_symbol)
        atoms = self.remove_adsorbate_from_slab(atoms, ads_index, ads_symbol)
        ads_pools = self.ads_pools.copy()
        ads_pools.remove(ads_symbol)
        random.seed(random.randint(1,100000000))
        ads_symbol = random.choice(ads_pools)
        atoms, _ = self.add_adsorbate_onto_slab(atoms, site_pos, ads_symbol, cutoff=self.tc_cutoff)
        return atoms

    def sort_by_position(self, atoms):
        """Sort atoms by positions."""
        atoms = copy.deepcopy(atoms)
        ps = atoms.get_positions()
        ps = ps.tolist()
        sorted_ps = sorted([(p, i) for i, p in enumerate(ps)])
        indices = [i for p, i in sorted_ps]
        return atoms[indices]
    
    def check_X_too_close(self, atoms, newp, oldp, cutoff, mic=True):
        """Check if there are atoms that are too close to each other.
        """
        newps = np.repeat(newp, len(oldp), axis=0)
        oldps = np.tile(oldp, (len(newp), 1))
        if mic:
            _, dists = find_mic(newps - oldps, atoms.cell, pbc=True)
        else:
            dists = np.linalg.norm(newps - oldps, axis=1)
        return any(dists < cutoff)
    
    def remove_too_close_X(self, atoms):
        """Chech if X is too close"""
        atoms = copy.deepcopy(atoms)
        cutoff = 0.4
        indices = [atom.index for atom in atoms]
        del_indices = []
        times = 1 # only check once
        for _ in range(times):
            for index in indices:
                a = copy.deepcopy(atoms)
                newp = a.positions[[index]]
                del a[index]
                oldp = a.positions
                if_too_close = self.check_X_too_close(a, newp, oldp, cutoff, mic=True)
                if if_too_close:
                    # print(f'deleting {index}')
                    del_indices.append(index)
                    break
            indices = [i for i in indices if i not in del_indices]
        atoms = atoms[indices]
        atoms = self.sort_by_position(atoms)
        return atoms
    
    def remove_atoms_outside_cell(self, atoms):
        """Remove_atoms outside cell"""
        a2 = atoms.copy()
        a2.wrap()
        indices = []
        for i in range(len(atoms)):
            if all(abs(atoms[i].position - a2[i].position) < 0.0001):
                indices.append(atoms[i].index)
        atoms = atoms[indices]
        atoms = self.sort_by_position(atoms)
        return atoms
  
    def add_full_adsorbates(self, atoms, sites, adsorbate='X'):
        """Add adsorbates on given sites"""
        distance = 0.0
        for site in sites:
            if adsorbate == 'X':
                ads = Atoms([Atom('X', (0, 0, 0))])
                ads.translate(site + (0., 0., distance))
            atoms.extend(ads)
        atoms = self.remove_atoms_outside_cell(atoms)
        return atoms
    
    def get_top_sites(self, points):
        """Get ontop sites"""
        top_sites = list(points)
        return top_sites
    
    def get_hollow_sites(self, tri_points):
        """Get hollow sites"""
        hollow_sites = []
        for tri_point in tri_points:
            average_site = np.average(tri_point, axis=0)
            hollow_sites.append(average_site)
        return hollow_sites

    def get_extended_points(self, atoms):
        """Get extended triangle points, return a numpy array"""
        cutoff = 6.0
        nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms),
                                self_interaction=True,
                                bothways=True,
                                skin=0.)
        nl.update(atoms)
        all_atoms_pos = []
        for atom in atoms:
            indices, offsets = nl.get_neighbors(atom.index)
            for i, offset in zip(indices, offsets):
                pos = atoms.positions[i] + np.dot(offset, atoms.get_cell())
                all_atoms_pos.append(tuple(pos))
        unique_atoms_pos = set(all_atoms_pos)
        unique_atoms_pos = np.vstack([np.array(unique_atom_pos) for unique_atom_pos in unique_atoms_pos])
        return unique_atoms_pos
    
    def get_occupied_sites(self, atoms_old, atoms_X, mic=True):
        """Get occupied sites"""
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms_old)
        X_indices, X_poss = [], []
        for atom in atoms_X:
            X_indices.append(atom.index)
            X_poss.append(atom.position)
        X_poss = np.asarray(X_poss)
        occupied_sites, occupied_adss = [], []
        if ads_indices != [] and ads_symbols != []:
            for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
                site_pos = self.get_ads_pos(atoms_old, ads_index, ads_symbol)
                if ads_symbol == 'CO' or ads_symbol == 'H' or ads_symbol == 'OH':
                    site_pos = site_pos + (0.0, 0.0, 2.0)
                else:
                    print('unsupported adsorbates')
                    assert False
                site_poss = np.tile(site_pos, (len(atoms_X), 1)) # tile n
                if mic:
                    _, dists = find_mic(site_poss - X_poss, atoms_old.cell, pbc=True)
                else:
                    dists = np.linalg.norm(site_poss - X_poss, axis=1)
                dists = dists.tolist()
                min_dist = min(dists)
                min_dist_index = dists.index(min_dist)
                occupied_sites.append(X_indices[min_dist_index])
                occupied_adss.append(ads_symbol)
        all_sites = X_indices
        return all_sites, occupied_sites, occupied_adss
        
    def get_sites(self, atoms):
        """Get adsorption sites for PdH(111) 2x2, and return adsorbates site atoms 
        and all site indices and corresponding occupied sites indices
        """
        atoms = atoms.copy()
        atoms_old = copy.deepcopy(atoms)
        atoms = atoms[[atom.index for atom in atoms if atom.tag==1]] # 1st layer Pd
        assert len(atoms) == 4
        extended_points = self.get_extended_points(atoms)
        points = extended_points
        tri = Delaunay(points[:,:2])
        simplices = tri.simplices
        tri_points = points[simplices]
        sites = {}
        sites['top'] = self.get_top_sites(points)
        sites['hollow'] = self.get_hollow_sites(tri_points)
        atoms = self.add_full_adsorbates(atoms, sites['top'], adsorbate='X')
        atoms = self.add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
        all_possible_sites = [atom.index for atom in atoms if atom.symbol == 'X']
        atoms_X = atoms[all_possible_sites]
        atoms_X = self.remove_too_close_X(atoms_X)
        all_sites, occupied_sites, occupied_adss = self.get_occupied_sites(atoms_old, atoms_X, mic=True)
        return atoms_X, all_sites, occupied_sites, occupied_adss

class AdsorbateSubstitution(AdsorbateOperator):
    """Adsorbate substitution: the number of adsorbates keeps unchanged and 
    substitute absorbate species"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 ads_pools=None, # add adsorbate pools:)
                 rng=np.random,
                 tc_cutoff=1.6):
        AdsorbateOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              ads_pools=ads_pools, # :)
                              rng=rng,
                              tc_cutoff=tc_cutoff)

        self.descriptor = 'AdsorbateSubstitution'
        self.ads_pools = ads_pools
        self.tc_cutoff = tc_cutoff

    def get_new_individual(self, parents):
        f = parents[0]
        flag = 0
        if bool(self.ads_pools):
            list1 = [a.tag for a in f if a.tag < 0] # non ads for tag setting
            if bool(list1):
               flag = 1
       # Permutation only makes sense if two different elements are present
        if flag == 0:
            f = parents[1]
            list1 = [a.tag for a in f if a.tag < 0]
            if not bool(list1):
                return None, '{1} not possible in {0}'.format(f.info['confid'],
                                                              self.descriptor)
        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [f.info['confid']]
        try:
            _, _ = self.get_adsorbates_from_slab(indi)
        except:
            from ase.visualize import view
            view(indi)
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        ads_indices, ads_symbols = [], []
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        assert len(ads_indices) == len(ads_symbols)
        print('======start AdsorbateSubstitution======')
        for _ in range(self.num_muts):
            if ads_indices != [] and ads_symbols != []:
                random.seed(random.randint(1,100000000))
                random_index = random.choice(range(len(ads_indices)))
                ads_index = ads_indices[random_index]
                ads_symbol = ads_symbols[random_index]
                atoms = self.substitute_adsorbate_on_slab(atoms, ads_index, ads_symbol)
                ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
            else:
                print(f'failed due to empty adsorbate.')
        _, _ = self.get_adsorbates_from_slab(atoms)
        print('======end AdsorbateSubstitution======')
        return atoms
    
class AdsorbateAddition(AdsorbateOperator):
    """Adsorbate addition: add adsorbate onto a slab"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 ads_pools=None, # add adsorbate pools:)
                 rng=np.random,
                 tc_cutoff=1.6):
        AdsorbateOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              ads_pools=ads_pools, # :)
                              rng=rng,
                              tc_cutoff=tc_cutoff)

        self.descriptor = 'AdsorbateAddition'
        self.ads_pools = ads_pools
        self.tc_cutoff = tc_cutoff

    def get_new_individual(self, parents):
        f = parents[0]
        flag = 0
        if bool(self.ads_pools):
            list1 = [a.tag for a in f if a.tag < 0] # non ads for tag setting
            if bool(list1):
               flag = 1
       # Permutation only makes sense if two different elements are present
        if flag == 0:
            f = parents[1]
            list1 = [a.tag for a in f if a.tag < 0]
            if not bool(list1):
                return None, '{1} not possible in {0}'.format(f.info['confid'],
                                                              self.descriptor)
        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [f.info['confid']]
        try:
            _, _ = self.get_adsorbates_from_slab(indi)
        except:
            view(indi)
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        assert len(ads_indices) == len(ads_symbols)
        print('======start AdsorbateAddition======')
        for _ in range(self.num_muts):
            atoms_old = copy.deepcopy(atoms)
            atoms_X, all_sites, occupied_sites, _ = self.get_sites(atoms_old)
            unoccupied_sites = [site for site in all_sites if site not in occupied_sites]
            if_too_close = True
            count = len(all_sites)
            while if_too_close:
                if unoccupied_sites == [] or count == 0:
                    print(f'failed due to too many adsorbate already on slab.')
                    break
                random.seed(random.randint(1,100000000))
                random_sites = random.choice(unoccupied_sites)
                unoccupied_sites.remove(random_sites)
                ads_symbol = random.choice(self.ads_pools)
                site_pos = atoms_X[random_sites].position
                atoms, if_too_close = self.add_adsorbate_onto_slab(atoms, site_pos, ads_symbol, cutoff=self.tc_cutoff)
                _, _ = self.get_adsorbates_from_slab(atoms)
                count -= 1
        print('======end AdsorbateAddition======')
        return atoms
    
class AdsorbateRemoval(AdsorbateOperator):
    """Adsorbate Remove: remove adsorbate from a slab"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 ads_pools=None, # add adsorbate pools:)
                 rng=np.random,
                 tc_cutoff=1.6):
        AdsorbateOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              ads_pools=ads_pools, # :)
                              rng=rng,
                              tc_cutoff=tc_cutoff)

        self.descriptor = 'AdsorbateRemoval'
        self.ads_pools = ads_pools
        self.tc_cutoff = tc_cutoff

    def get_new_individual(self, parents):
        f = parents[0]
        flag = 0
        if bool(self.ads_pools):
            list1 = [a.tag for a in f if a.tag < 0] # non ads for tag setting
            if bool(list1):
               flag = 1
       # Permutation only makes sense if two different elements are present
        if flag == 0:
            f = parents[1]
            list1 = [a.tag for a in f if a.tag < 0]
            if not bool(list1):
                return None, '{1} not possible in {0}'.format(f.info['confid'],
                                                              self.descriptor)
        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [f.info['confid']]
        try:
            _, _ = self.get_adsorbates_from_slab(indi)
        except:
            view(indi)
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        assert len(ads_indices) == len(ads_symbols)
        print('======start AdsorbateRemoval======')
        for _ in range(self.num_muts):
            if ads_indices != [] and ads_symbols != []:
                random.seed(random.randint(1,100000000))
                random_index = random.choice(range(len(ads_indices)))
                ads_index = ads_indices[random_index]
                ads_symbol = ads_symbols[random_index]
                atoms = self.remove_adsorbate_from_slab(atoms, ads_index, ads_symbol)
                ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
            else:
                print(f'failed due to empty adsorbate.')
        print('======end AdsorbateRemoval======')
        return atoms

class AdsorbateSwapOccupied(AdsorbateOperator):
    """Adsorbate swap: swap two occupied adsorbates on a slab"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 ads_pools=None, # add adsorbate pools:)
                 rng=np.random,
                 tc_cutoff=1.6):
        AdsorbateOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              ads_pools=ads_pools, # :)
                              rng=rng,
                              tc_cutoff=tc_cutoff)

        self.descriptor = 'AdsorbateSwapOccupied'
        self.ads_pools = ads_pools
        self.tc_cutoff = tc_cutoff

    def get_new_individual(self, parents):
        f = parents[0]
        flag = 0
        if bool(self.ads_pools):
            list1 = [a.tag for a in f if a.tag < 0] # non ads for tag setting
            if bool(list1):
               flag = 1
       # Permutation only makes sense if two different elements are present
        if flag == 0:
            f = parents[1]
            list1 = [a.tag for a in f if a.tag < 0]
            if not bool(list1):
                return None, '{1} not possible in {0}'.format(f.info['confid'],
                                                              self.descriptor)
        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [f.info['confid']]
        try:
            _, _ = self.get_adsorbates_from_slab(indi)
        except:
            view(indi)
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        assert len(ads_indices) == len(ads_symbols)
        print('======start Swap======')
        
        for _ in range(self.num_muts):
            if len(ads_indices)>=2 and len(set(ads_symbols))>=2:  
                random.seed(random.randint(1,100000000))
                random_index1 = random.choice(range(len(ads_indices)))
                ads_index1 = ads_indices[random_index1]
                ads_symbol1 = ads_symbols[random_index1]
                site_pos1 = self.get_ads_pos(atoms, ads_index1, ads_symbol1)
                atoms = self.remove_adsorbate_from_slab(atoms, ads_index1, ads_symbol1)
                ads_indices2, ads_symbols2 = self.get_adsorbates_from_slab(atoms)
                ads2_indices, ads2_symbols = [], []
                for ads_index, ads_symbol in zip(ads_indices2, ads_symbols2):
                    if ads_symbol != ads_symbol1:
                        ads2_indices.append(ads_index)
                        ads2_symbols.append(ads_symbol)
                if ads2_indices != [] and ads2_symbols != []:
                    random.seed(random.randint(1,100000000))
                    random_index2 = random.choice(range(len(ads2_indices)))
                    ads_index2 = ads2_indices[random_index2]
                    ads_symbol2 = ads2_symbols[random_index2]
                site_pos2 = self.get_ads_pos(atoms, ads_index2, ads_symbol2)
                atoms = self.remove_adsorbate_from_slab(atoms, ads_index2, ads_symbol2)
                _, _ = self.get_adsorbates_from_slab(atoms)
                atoms, _ = self.add_adsorbate_onto_slab(atoms, site_pos1, ads_symbol2, cutoff=self.tc_cutoff)
                atoms, _ = self.add_adsorbate_onto_slab(atoms, site_pos2, ads_symbol1, cutoff=self.tc_cutoff)
                ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        else:
            print(f'failed due to too less adsorbate.')
        print('======end Swap======')
        return atoms
    
class AdsorbateMoveToUnoccupied(AdsorbateOperator):
    """Adsorbate swap: swap two occupied adsorbates on a slab"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 ads_pools=None, # add adsorbate pools:)
                 rng=np.random,
                 tc_cutoff=1.6):
        AdsorbateOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              ads_pools=ads_pools, # :)
                              rng=rng,
                              tc_cutoff=tc_cutoff)

        self.descriptor = 'AdsorbateMoveToUnoccupied'
        self.ads_pools = ads_pools
        self.tc_cutoff = tc_cutoff

    def get_new_individual(self, parents):
        f = parents[0]
        flag = 0
        if bool(self.ads_pools):
            list1 = [a.tag for a in f if a.tag < 0] # non ads for tag setting
            if bool(list1):
               flag = 1
       # Permutation only makes sense if two different elements are present
        if flag == 0:
            f = parents[1]
            list1 = [a.tag for a in f if a.tag < 0]
            if not bool(list1):
                return None, '{1} not possible in {0}'.format(f.info['confid'],
                                                              self.descriptor)
        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [f.info['confid']]
        try:
            _, _ = self.get_adsorbates_from_slab(indi)
        except:
            view(indi)
            print('get_adsorbates_from_slab error')
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        assert len(ads_indices) == len(ads_symbols)
        print('======start MoveToUnoccupied======')
        for _ in range(self.num_muts):
            if ads_indices != []:
                atoms = copy.deepcopy(atoms)
                atoms_old = copy.deepcopy(atoms)
                atoms_X, all_sites, occupied_sites, _ = self.get_sites(atoms_old)
                unoccupied_sites = [site for site in all_sites if site not in occupied_sites]
                random.seed(random.randint(1,100000000))
                random_index = random.choice(range(len(ads_indices)))
                ads_index = ads_indices[random_index]
                ads_symbol = ads_symbols[random_index]
                atoms = self.remove_adsorbate_from_slab(atoms, ads_index, ads_symbol)
                if_too_close = True
                count = len(all_sites)
                while if_too_close:
                    random.seed(random.randint(1,100000000))
                    random_sites = random.choice(unoccupied_sites)
                    site_pos = atoms_X[random_sites].position
                    atoms, if_too_close = self.add_adsorbate_onto_slab(atoms, site_pos, ads_symbol, cutoff=self.tc_cutoff)
                    _, _ = self.get_adsorbates_from_slab(atoms)
                    count -= 1
                    if count == 0:
                        print(f'failed due to too many adsorbate already on slab.')
                        break
        print('======end MoveToUnoccupied======')
        return atoms

class AdsorbateCutSpliceCrossover(AdsorbateOperator):
    """Adsorbate swap: swap two occupied adsorbates on a slab"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 ads_pools=None, # add adsorbate pools:)
                 rng=np.random,
                 tc_cutoff=1.6):
        AdsorbateOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              ads_pools=ads_pools, # :)
                              rng=rng,
                              tc_cutoff=tc_cutoff)

        self.descriptor = 'AdsorbateCutSpliceCrossover'
        self.ads_pools = ads_pools
        self.tc_cutoff = tc_cutoff

    def get_new_individual(self, parents):
        
        f = parents[0]
        m =  parents[1]
        
        indi1 = self.initialize_individual(f, f)
        indi1.info['data'] = f.info['data']
        indi1.info['data']['parents'] = [i.info['confid'] for i in parents]
        
        indi2 = self.initialize_individual(m, m)
        indi2.info['data'] = m.info['data']
        indi2.info['data']['parents'] = [i.info['confid'] for i in parents]
        try:
            _, _ = self.get_adsorbates_from_slab(indi1)
            _, _ = self.get_adsorbates_from_slab(indi2)
        except:
            view(indi1)
            view(indi2)
        indi = self.operate(indi1, indi2)
        parent_message = ': Parents {0} {1}'.format(f.info['confid'],
                                                    m.info['confid'])
        if indi == None:
            return None, self.descriptor + parent_message
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms1, atoms2):
        # Do the operation
        ads_indices1, ads_symbols1 = self.get_adsorbates_from_slab(atoms1)
        ads_indices2, ads_symbols2 = self.get_adsorbates_from_slab(atoms2)
        print('======start AdsorbateCutSpliceCrossover======')
        for _ in range(self.num_muts):
            atoms_old1 = copy.deepcopy(atoms1)
            atoms_old2 = copy.deepcopy(atoms2)
            atoms_X1, all_sites1, occupied_sites1, occupied_adss1 = self.get_sites(atoms_old1)
            atoms_X2, all_sites2, occupied_sites2, occupied_adss2 = self.get_sites(atoms_old2)
            occupied_sites2_from_sites1, occupied_adss2_from_adss1 = [], []
            # print(all_sites1, all_sites2)
            assert len(all_sites1)==len(all_sites2)
            for occupied_site1, occupied_ads1 in zip(occupied_sites1, occupied_adss1):
                if occupied_site1 in all_sites1:
                    occupied_index1 = all_sites1.index(occupied_site1)
                    occupied_sites2_from_sites1.append(all_sites2[occupied_index1])
                    occupied_adss2_from_adss1.append(occupied_ads1)
                else:
                    print('Occupied site not in all sites. Something error')
                    assert False
            atoms = atoms_old2[[atom.index for atom in atoms_old2 if atom.tag>=0]] # remove all adss
            atoms.info['data']['ads_indices'] = []
            atoms.info['data']['ads_symbols'] = []
            
            for occupied_site, occupied_ads in zip(occupied_sites2_from_sites1, occupied_adss2_from_adss1):
                site_pos = atoms_X2[occupied_site].position
                ads_symbol = occupied_ads
                atoms, if_too_close = self.add_adsorbate_onto_slab(atoms, site_pos, ads_symbol, cutoff=self.tc_cutoff)
                if if_too_close:
                    print('fail AdsorbateCutSpliceCrossover')
                    return None
        print('======end AdsorbateCutSpliceCrossover======')
        return atoms
    
class InteranlHydrogenOperator(OffspringCreator):
    """Adsorbate operator"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 internal_H_pools=['H'],
                 rng=np.random):
        OffspringCreator.__init__(self, verbose, num_muts=num_muts, rng=rng)

        self.allowed_compositions = allowed_compositions
        self.internal_H_pools = internal_H_pools
        
    def atoms_too_close_after_addition(self, atoms, n_added, cutoff, mic=True): 
        """Check if there are atoms that are too close to each other after 
        adding some new atoms.

        n_added : int
            Number of newly added atoms.
        cutoff : float, default 1.5
            The cutoff radius. Two atoms are too close if the distance between
            them is less than this cutoff
        mic : bool, default False
            Whether to apply minimum image convention. Remember to set 
            mic=True for periodic systems.
        """
        newp, oldp = atoms.positions[-n_added:], atoms.positions[:-n_added]
        newps = np.repeat(newp, len(oldp), axis=0)
        oldps = np.tile(oldp, (n_added, 1))
        if mic:
            _, dists = find_mic(newps - oldps, atoms.cell, pbc=True)
        else:
            dists = np.linalg.norm(newps - oldps, axis=1)
        return any(dists < cutoff)
            
    def debug_atoms_data(self, atoms, ads_indices, ads_symbols):
        print('===debug start===')
        for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
            print(str(atoms[ads_index].symbols), ads_symbol)
        print('---ads location---')
        for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
            print(str(atoms[ads_index].symbols), ads_symbol)
            for i in ads_index:
                s = atoms[i].symbol
                assert (s in ads_symbol)
        print('===debug end===')
        return True
    
    def get_adsorbates_from_slab(self, atoms, debug=False):
        """Get adsorbate information from atoms, including indices and symbols"""
        ads_indices = atoms.info['data']['ads_indices']
        ads_symbols = atoms.info['data']['ads_symbols']
        assert len(ads_indices)==len(ads_symbols)
        if debug:
            self.debug_atoms_data(atoms, ads_indices, ads_symbols)
        return ads_indices, ads_symbols
    
    def update_all_adss_indices(self, atoms, atoms_old):
        """Update adsorbates indices after deleting one adsorbate"""
        update_ads_indices = []
        update_ads_symbols = []
        ads_indices_old, ads_symbols_old = self.get_adsorbates_from_slab(atoms_old, debug=False)
        if len(atoms) < len(atoms_old): # due to deleting adsorbate
            for ads_index_old, ads_symbol_old in zip(ads_indices_old, ads_symbols_old):
                update_ads_index = []
                for atom_index_old in ads_index_old:
                    atom_pos_old = atoms_old[atom_index_old].position
                    atoms_sym_old = atoms_old[atom_index_old].symbol
                    ads_indices_new = [atom.index for atom in atoms if atom.tag<0]
                    for atom_index_new in ads_indices_new:
                        atom_pos_new = atoms[atom_index_new].position
                        atoms_sym_new = atoms[atom_index_new].symbol
                        if (abs(atom_pos_new-atom_pos_old)<0.0000001).all() and atoms_sym_new==atoms_sym_old:
                            update_ads_index.append(atom_index_new)
                if update_ads_index != [] and len(update_ads_index)==len(ads_index_old):
                    update_ads_indices.append(update_ads_index)
                    update_ads_symbols.append(ads_symbol_old)
        return update_ads_indices, update_ads_symbols
    
    def remove_atoms_outside_cell(self, atoms):
        """Remove_atoms outside cell"""
        a2 = atoms.copy()
        a2.wrap()
        indices = []
        for i in range(len(atoms)):
            if all(abs(atoms[i].position - a2[i].position) < 0.0001):
                indices.append(atoms[i].index)
        atoms = atoms[indices]
        return atoms
    
    def add_full_adsorbates(self, atoms, sites, adsorbate='X'):
        """Add adsorbates on given sites"""
        atoms = atoms.copy()
        distance = 0.0
        for site in sites:
            if adsorbate == 'X':
                ads = Atoms([Atom('X', (0, 0, 0))])
                ads.translate(site + (0., 0., distance))
            atoms.extend(ads)
        atoms = self.remove_atoms_outside_cell(atoms)
        return atoms
    
    def get_top_sites(self, points):
        """Get ontop sites"""
        top_sites = list(points)
        return top_sites
    
    def get_hollow_sites(self, tri_points):
        """Get hollow sites"""
        hollow_sites = []
        for tri_point in tri_points:
            average_site = np.average(tri_point, axis=0)
            hollow_sites.append(average_site)
        return hollow_sites

    def get_extended_points(self, atoms):
        """Get extended triangle points, return a numpy array"""
        cutoff = 6.0
        nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms),
                                self_interaction=True,
                                bothways=True,
                                skin=0.)
        nl.update(atoms)
        all_atoms_pos = []
        for atom in atoms:
            indices, offsets = nl.get_neighbors(atom.index)
            for i, offset in zip(indices, offsets):
                pos = atoms.positions[i] + np.dot(offset, atoms.get_cell())
                all_atoms_pos.append(tuple(pos))
        unique_atoms_pos = set(all_atoms_pos)
        unique_atoms_pos = np.vstack([np.array(unique_atom_pos) for unique_atom_pos in unique_atoms_pos])
        return unique_atoms_pos
    
    def get_ith_layer_all_H_sites(self, atoms, atomsX_hollow_current, atomsX_top_next, mic=True):
        """Get occupied sites"""
        atoms = atoms.copy()
        assert len(atomsX_top_next)==4
        indices_current, poss_current = [], []
        for atom in atomsX_hollow_current:
            indices_current.append(atom.index)
            poss_current.append(atom.position)
        poss_current = np.asarray(poss_current)
        
        X_top_next_poss = atomsX_top_next.positions
        Hs_indices = []
        for pos_next in X_top_next_poss:
            poss_next = np.tile(pos_next, (len(atomsX_hollow_current), 1)) # tile n
            if mic:
                _, dists = find_mic(poss_current - poss_next, atoms.cell, pbc=True)
            else:
                dists = np.linalg.norm(poss_current - poss_next, axis=1)
            dists = dists.tolist()
            min_dist = min(dists)
            min_dist_index = dists.index(min_dist)
            Hs_indices.append(indices_current[min_dist_index])
        
        atoms_all_Hs = atomsX_hollow_current[Hs_indices] # all possible H atoms in this layer
        assert len(atoms_all_Hs)==4
        return atoms_all_Hs
    
    def get_ith_layer_metal_sites(self, atoms, i=3):
        """Get sites of internal H in the slab, here are 2nd and 3rd H layers"""
        atoms = atoms.copy()
        atoms = atoms[[atom.index for atom in atoms if atom.tag==i]] # 3st layer Pd
        assert len(atoms) == 4
        extended_points = self.get_extended_points(atoms)
        points = extended_points
        tri = Delaunay(points[:,:2])
        simplices = tri.simplices
        tri_points = points[simplices]
        sites = {}
        sites['top'] = self.get_top_sites(points)
        sites['hollow'] = self.get_hollow_sites(tri_points)
        atoms_with_top = self.add_full_adsorbates(atoms, sites['top'], adsorbate='X')
        atoms_with_hollow = self.add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
        atomsX_top = atoms_with_top[[atom.index for atom in atoms_with_top if atom.symbol == 'X']]
        atomsX_hollow = atoms_with_hollow[[atom.index for atom in atoms_with_hollow if atom.symbol == 'X']]
        return atomsX_top, atomsX_hollow
    
    def get_occupied_Hs(self, atoms_base, atoms_ads, mic=True):
        """Get occupied H atoms from all H atoms"""
        assert len(atoms_base)==4
        indices_ads, poss_ads = [], []
        for atom in atoms_ads:
            indices_ads.append(atom.index)
            poss_ads.append(atom.position)
        poss_ads = np.asarray(poss_ads)
        
        indices_base, poss_base = [], []
        for atom in atoms_base:
            indices_base.append(atom.index)
            poss_base.append(atom.position)
        poss_base = np.asarray(poss_base)
        occupied_indices = []
        for i, pos_ads in enumerate(poss_ads):
            poss_ads_tile = np.tile(pos_ads, (len(atoms_base), 1)) # tile n
            if mic:
                _, dists = find_mic(poss_ads_tile - poss_base, atoms_base.cell, pbc=mic)
            else:
                dists = np.linalg.norm(poss_ads_tile - poss_base, axis=1)
            dists = dists.tolist()
            min_dist = min(dists)
            min_dist_index = dists.index(min_dist)
            occupied_indices.append(indices_base[min_dist_index])
        atoms_X = atoms_base
        all_indices = indices_base
        assert len(occupied_indices)<=4
        return atoms_X, all_indices, occupied_indices
    
    def get_ith_layer_H_sites(self, atoms, layer=3):
        """Get H atoms sites of ith layer, site z is position z of ith layer. i ranges from top to bottom"""
        atoms = copy.deepcopy(atoms)
        _, atomsX_hollow_current = self.get_ith_layer_metal_sites(atoms, layer)
        atomsX_top_next, _ = self.get_ith_layer_metal_sites(atoms, layer+1) # (i+1)st layer Pd
        atoms_all_Hs = self.get_ith_layer_all_H_sites(atoms, atomsX_hollow_current, atomsX_top_next)
        indices_existed_Hs = [atom.index for atom in atoms if atom.tag in [layer+4]] # ist bilayer H
        assert len(indices_existed_Hs)<=4
        if indices_existed_Hs != []:
            atoms_existed_Hs = atoms[indices_existed_Hs] # ist layer H
            atoms_X, all_indices, occupied_indices = self.get_occupied_Hs(atoms_all_Hs, atoms_existed_Hs, mic=True)
        else:
            atoms_X = atoms_all_Hs
            all_indices = [atom.index for atom in atoms_X]
            occupied_indices = []
        return atoms_X, all_indices, occupied_indices
    
    def get_vacancies_Hs_atoms(self, atoms, bilayers=[2, 3]):
        """Automatically idendify X vaccavies atoms"""
        atoms = copy.deepcopy(atoms)
        atoms_vacancies = Atoms()
        for bilayer in bilayers:
            atoms_X, all_indices, occupied_indices = self.get_ith_layer_H_sites(atoms, layer=bilayer)
            unoccupied_indices = [index for index in all_indices if index not in occupied_indices]
            if unoccupied_indices != []:
                atoms_X_unoccupied = atoms_X[unoccupied_indices]
                z_average = np.mean([atom.z for atom in atoms if atom.tag==bilayer or atom.tag==bilayer-1])
                for atom in atoms_X_unoccupied:
                    atom.z =  z_average
                    atom.tag = bilayer+4
                atoms_vacancies.extend(atoms_X_unoccupied)
        return atoms_vacancies
    
    def get_existed_Hs_indices(self, atoms,  bilayers=[2, 3]):
        """Get existed H atoms in slab"""
        atoms = copy.deepcopy(atoms)
        all_existed_Hs_indices = []
        for bilayer in bilayers:
            indices_existed_Hs = [atom.index for atom in atoms if atom.tag in [bilayer+4]] # ist bilayer H
            all_existed_Hs_indices += indices_existed_Hs
        return all_existed_Hs_indices


class InternalHydrogenAddition(InteranlHydrogenOperator):
    """Adsorbate addition: add adsorbate onto a slab"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 internal_H_pools=None, # add adsorbate pools:)
                 rng=np.random):
        InteranlHydrogenOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              internal_H_pools=internal_H_pools, # :)
                              rng=rng)

        self.descriptor = 'InternalHydrogenAddition'
        self.internal_H_pools = internal_H_pools

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [f.info['confid']]
        try:
            _, _ = self.get_adsorbates_from_slab(indi)
        except:
            view(indi)
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        assert len(ads_indices) == len(ads_symbols)
        print('======start HydrogenAddition======')
        for _ in range(self.num_muts):
            atoms = copy.deepcopy(atoms)
            atoms_old = copy.deepcopy(atoms)
            atoms_vacancies = self.get_vacancies_Hs_atoms(atoms, bilayers=[2, 3])
            if len(atoms_vacancies)!=0:
                indices_X = [atom.index for atom in atoms_vacancies]
                random.seed(random.randint(1,100000000))
                random_index = random.choice(indices_X)
                random_atom_X = atoms_vacancies[random_index]
                random_atom_X.symbol = 'H'
                print('adding one internal H')
                atoms.extend(random_atom_X)
            else:
                atoms = atoms_old
        print('======end HydrogenAddition======')
        return atoms
    
class InternalHydrogenRemoval(InteranlHydrogenOperator):
    """Adsorbate addition: add adsorbate onto a slab"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 internal_H_pools=None, # add adsorbate pools:)
                 rng=np.random):
        InteranlHydrogenOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              internal_H_pools=internal_H_pools, # :)
                              rng=rng)

        self.descriptor = 'InternalHydrogenRemoval'
        self.internal_H_pools = internal_H_pools

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [f.info['confid']]
        try:
            _, _ = self.get_adsorbates_from_slab(indi)
        except:
            view(indi)
            assert False
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        assert len(ads_indices) == len(ads_symbols)
        print('======start InternalHydrogenRemoval======')
        for _ in range(self.num_muts):
            atoms = copy.deepcopy(atoms)
            atoms_old = copy.deepcopy(atoms)
            all_existed_Hs_indices = self.get_existed_Hs_indices(atoms,  bilayers=[2, 3])
            if len(all_existed_Hs_indices)!=0:
                random.seed(random.randint(1,100000000))
                random_index = random.choice(all_existed_Hs_indices)
                del atoms[random_index]
                update_ads_indices, update_ads_symbols = self.update_all_adss_indices(atoms, atoms_old)
                atoms.info['data']['ads_symbols'] = update_ads_symbols
                atoms.info['data']['ads_indices'] = update_ads_indices
                print('removing one internal H')
            else:
                atoms = atoms_old         
        print('======end InternalHydrogenRemoval======')
        return atoms

class InternalHydrogenMoveToUnoccupied(InteranlHydrogenOperator):
    """Adsorbate addition: add adsorbate onto a slab"""
    def __init__(self, verbose=False, num_muts=1,
                 allowed_compositions=None,
                 internal_H_pools=None, # add adsorbate pools:)
                 rng=np.random):
        InteranlHydrogenOperator.__init__(self, verbose, num_muts,
                              allowed_compositions,
                              internal_H_pools=internal_H_pools, # :)
                              rng=rng)

        self.descriptor = 'InternalHydrogenMoveToUnoccupied'
        self.internal_H_pools = internal_H_pools

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.initialize_individual(f, f)
        indi.info['data'] = f.info['data']
        indi.info['data']['parents'] = [f.info['confid']]
        try:
            _, _ = self.get_adsorbates_from_slab(indi)
        except:
            view(indi)
            assert False
        indi = self.operate(indi)
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi),
                self.descriptor + parent_message)

    def operate(self, atoms):
        # Do the operation
        ads_indices, ads_symbols = self.get_adsorbates_from_slab(atoms)
        assert len(ads_indices) == len(ads_symbols)
        print('======start InternalHydrogenMoveToUnoccupied======')
        for _ in range(self.num_muts):
            atoms = copy.deepcopy(atoms)
            atoms_old = copy.deepcopy(atoms)
            all_existed_Hs_indices = self.get_existed_Hs_indices(atoms,  bilayers=[2, 3])
            atoms_vacancies = self.get_vacancies_Hs_atoms(atoms, bilayers=[2, 3])
            if len(atoms_vacancies)!=0 and len(all_existed_Hs_indices)!=0:
                random.seed(random.randint(1,100000000))
                random_index = random.choice(all_existed_Hs_indices)
                del atoms[random_index]
                update_ads_indices, update_ads_symbols = self.update_all_adss_indices(atoms, atoms_old)
                atoms.info['data']['ads_symbols'] = update_ads_symbols
                atoms.info['data']['ads_indices'] = update_ads_indices
                print('removing one internal H')
                
                indices_X = [atom.index for atom in atoms_vacancies]
                random.seed(random.randint(1,100000000))
                random_index = random.choice(indices_X)
                random_atom_X = atoms_vacancies[random_index]
                random_atom_X.symbol = 'H'
                print('adding one internal H')
                atoms.extend(random_atom_X)
            else:
                atoms = atoms_old         
        print('======end InternalHydrogenMoveToUnoccupied======')
        return atoms
