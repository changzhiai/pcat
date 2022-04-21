from contextlib import contextmanager
from ase.io import read, write
import numpy as np
import os
import subprocess
import time
from ase.db import connect

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


submit = '#!/bin/bash\n#SBATCH --mail-user=changai@dtu.dk\n#SBATCH --mail-type=ALL\n#SBATCH -N 1\n#SBATCH -n 24\n#SBATCH --time=50:00:00\n###SBATCH --mem=4G  #4 GB RAM per node\n#SBATCH --output=mpi_job_slurm.log\n#SBATCH --job-name=job_name\n#SBATCH --partition=xeon24\nexport CPU_ARCH=broadwell\nmodule use /home/energy/modules/modules/all # added into env variable\nmodule load VASP/5.4.4-intel-2019b\nmodule load ASE/3.20.1-intel-2019b-Python-3.7.4\nexport ASE_VASP_VDW=/home/energy/modules/software/VASP/vasp-potpaw-5.4\nexport VASP_PP_PATH=/home/energy/modules/software/VASP/vasp-potpaw-5.4/\nexport ASE_VASP_COMMAND="mpiexec -n 24 vasp_std"  #for Vasp\n./adsorption_energy.py arg > results'

db = connect('dft_candidates_PdHx_r6.db')
sites = ['top1', 'hollow1']
adsorbates = ['surface', 'HOCO', 'CO', 'H', 'OH']

for row in db.select(calc='clease'):
    atoms = row.toatoms()
    name = row.get('formula')
    row_id = row.get('id')
    if row_id == 2:
        continue
    with cd('{0}_{1}'.format(row_id, name)):
        for site in sites:
            with cd('{0}'.format(site)):
                for adsorbate in adsorbates:
                    if adsorbate == 'surface' and site != 'top1':
                        continue
                    with cd('{0}'.format(adsorbate)):
                        # os.system('rm *')
                        job_id = str(row_id) + '_' + name + '_' + site + '_' + adsorbate
                        time.sleep(1.)
                        write('POSCAR-start', atoms)
                        args = str(row_id) + ' ' + name + ' ' + site + ' ' + adsorbate
                        os.system('cp ../../../adsorption_energy.py .')
                        submit_txt = submit.replace('job_name', job_id)
                        submit_txt = submit_txt.replace('arg',args)
                        
                        submit_file = 'submit_'+name+'.sh'
                        o = open(submit_file,'w')
                        o.write(submit_txt)
                        o.close()
                        
                        print(job_id)
                        # subprocess.run(["sh",submit_file])
                        os.system('sbatch ' + submit_file)
    # break
