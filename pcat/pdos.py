import numpy as np
from ase.calculators.vasp import Vasp, VaspDos
import matplotlib.pyplot as plt
import matplotlib

class PDOS:
    def __init__(self, folder='scf'):
        self.folder = folder

    def plot(self, indices=[0]):
        "plot for d band, which is really easy to extend to other band"
        calc = Vasp(self.folder) 
        ados = VaspDos(efermi=calc.get_fermi_level())
        energy = ados.energy
        pdos_d_up = 0
        pdos_d_down = 0
        pdos_d = 0
        for i in [indices]: 
            for each in ['dxy+', 'dyz+', 'dz2+', 'dxz+', 'dx2+']:
                pdos_d_up = pdos_d_up + ados.site_dos(i, each)
            for each in ['dxy-', 'dyz-', 'dz2-', 'dxz-', 'dx2-']:
                pdos_d_down = pdos_d_down + ados.site_dos(i, each)
            for each in ['dxy+', 'dyz+', 'dz2+', 'dxz+', 'dx2+', 'dxy-', 'dyz-', 'dz2-', 'dxz-', 'dx2-']:
                pdos_d = pdos_d + ados.site_dos(i, each)

        center = np.trapz(pdos_d * energy, energy) / np.trapz(pdos_d, energy)
        width = np.sqrt(np.trapz(pdos_d * energy**2, energy) / np.trapz(pdos_d, energy))
        fig = plt.figure()
        plt.title('d-band center = %5.3f eV, d-band width = %5.3f eV' % (center, width), fontsize=12)
        plt.plot(energy, pdos_d_up, color='red', label='M atom d orbit')
        plt.plot(energy, -pdos_d_down, color='red')
        plt.xlabel('Energy (eV)')
        plt.xlim(-10, 10)
        plt.ylabel('PDOS')
        plt.legend()                
        fig.savefig('pdos.png')

if __name__ == '__main__':
    ''