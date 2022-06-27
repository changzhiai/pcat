from contextlib import contextmanager
import os

@contextmanager
def cd(newdir):
    """Create and go to the directory"""
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

@contextmanager
def walk(newdir):
    """Only go to the directory"""
    prevdir = os.getcwd()
    # try:
    #     os.makedirs(newdir)
    # except OSError:
    #     pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def submit(arg=None, cores=24):
    """Submit jobs for exon24"""
    submit =  '#!/bin/bash\n'
    submit += '#SBATCH --mail-user=changai@dtu.dk\n'
    submit += '#SBATCH --mail-type=ALL\n'
    submit += '#SBATCH -N 1\n'
    submit += '#SBATCH -n {0}\n'.format(cores)
    submit += '#SBATCH --time=49:00:00\n'
    submit += '###SBATCH --mem=4G  #4 GB RAM per node\n'
    submit += '#SBATCH --output=mpi_job_slurm.log\n'
    submit += '#SBATCH --job-name=job_name\n'
    submit += '#SBATCH --partition=xeon{0}\n'.format(cores)
    if cores == 16:
        submit += 'export CPU_ARCH=sandybridge\n'
    elif cores == 24:
        submit += 'export CPU_ARCH=broadwell\n'
    elif cores == 40:
        submit += 'export CPU_ARCH=skylake\n'
    elif cores == 56:
        submit += 'export CPU_ARCH=icelake\n'
    submit += 'module use /home/energy/modules/modules/all # added into env variable\n'
    submit += 'module load VASP/5.4.4-intel-2019b\n'
    submit += 'module load ASE/3.20.1-intel-2019b-Python-3.7.4\n'
    submit += 'export ASE_VASP_VDW=/home/energy/modules/software/VASP/vasp-potpaw-5.4\n'
    submit += 'export VASP_PP_PATH=/home/energy/modules/software/VASP/vasp-potpaw-5.4/\n'
    submit += 'export ASE_VASP_COMMAND="mpiexec -n {0} vasp_std"  #for Vasp\n'.format(cores)
    submit += './adsorption_energy.py {0} > results'.format(arg)
    return submit

def submit_cmd(cmd=None, cores=24):
    """Submit jobs for exon24"""
    submit =  '#!/bin/bash\n'
    submit += '#SBATCH --mail-user=changai@dtu.dk\n'
    submit += '#SBATCH --mail-type=ALL\n'
    submit += '#SBATCH -N 1\n'
    submit += '#SBATCH -n {0}\n'.format(cores)
    submit += '#SBATCH --time=49:00:00\n'
    submit += '###SBATCH --mem=4G  #4 GB RAM per node\n'
    submit += '#SBATCH --output=mpi_job_slurm.log\n'
    submit += '#SBATCH --job-name=job_name\n'
    submit += '#SBATCH --partition=xeon{0}\n'.format(cores)
    if cores == 16:
        submit += 'export CPU_ARCH=sandybridge\n'
    elif cores == 24:
        submit += 'export CPU_ARCH=broadwell\n'
    elif cores == 40:
        submit += 'export CPU_ARCH=skylake\n'
    elif cores == 56:
        submit += 'export CPU_ARCH=icelake\n'
    submit += 'module use /home/energy/modules/modules/all # added into env variable\n'
    submit += 'module load VASP/5.4.4-intel-2019b\n'
    submit += 'module load ASE/3.20.1-intel-2019b-Python-3.7.4\n'
    submit += 'export ASE_VASP_VDW=/home/energy/modules/software/VASP/vasp-potpaw-5.4\n'
    submit += 'export VASP_PP_PATH=/home/energy/modules/software/VASP/vasp-potpaw-5.4/\n'
    submit += 'export ASE_VASP_COMMAND="mpiexec -n {0} vasp_std"  #for Vasp\n'.format(cores)
    submit += cmd
    return submit