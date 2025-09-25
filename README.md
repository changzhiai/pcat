# PCAT: **P**ractical **CA**talysis **T**oolkit

Python toolkit package for analyzing, pre-processing and post-processing with density functional theory, cluster expansion, graph neural network, Monte Carlo simulated annealing, genetic algorithm, and active learning workflows in the field of catalysis. 


It includes support for advanced genetic algorithm operators and offers comprehensive analysis tools such as formation energy, free energy, 3D convex hull construction, scaling relations, activity volcano plots, Pourbaix diagrams, and projected density of states (PDOS). Some examples can be found in `examples` folder. Especally, active learning workflow can be found in [`workflow`](https://github.com/changzhiai/pcat/tree/master/instances/instance4_ml_ga/workflow).


## Installation

1. Clone the repository:

    `git clone https://gitlab.com/changzhiai/pcat.git`

2. Enter the installation path: 

    `cd pcat`

3. Install PCAT package:

   `python setup.py install`

## Publications using PCAT and how to cite

1. Density functional theory method with a kinetic model under a practical workflow was applied to screen doped Pd hydrides. The code can be found in the folder [`instances/instance1_dft`](https://github.com/changzhiai/pcat/tree/master/instances/instance1_dft) and [Jupyter-notebook](https://github.com/changzhiai/PlotPackage/blob/master/plotpackage/myproject/version3/Paper1_more_element.ipynb) and the details can be found in this paper:


    [Metal-doped PdH(111) catalysts for CO<sub>2</sub> reduction. Changzhi Ai, Tejs Vegge and Heine Anton Hansen, _ChemSusChem_ **2022**, 15, e202200008.](https://doi.org/10.1002/cssc.202200008)


2. Cluster expansion with Monte Carlo simulated annealing method was applied to study hydrogen impact on CO<sub>2</sub> reduction in PdH<sub>x</sub>. The code can be found in the folder [`instances/instance2_ce_mcsa`](https://github.com/changzhiai/pcat/tree/master/instances/instance2_ce_mcsa) and the details can be found in this paper: 

    [Impact of hydrogen concentration for reduction on PdH<sub>x</sub>: A combination study of cluster expansion and kinetics analysis. Changzhi Ai, Jin Hyun Chang, Alexander Sougaard Tygesen, Tejs Vegge and Heine Anton Hansen, _Journal of Catalysis_, **2023**, 428, 115188.](https://doi.org/10.1016/j.jcat.2023.115188)

3. Cluster expansion with Monte Carlo simulated annealing method was applied to high-throughput compositional screening of metal alloy hydride. The code can be found in the folder [`instances/instance3_ce_mcsa`](https://github.com/changzhiai/pcat/tree/master/instances/instance3_ce_mcsa) and the details can be found in this paper:

    [High-throughput compositional screening of Pd<sub>x</sub>Ti<sub>1-x</sub>H<sub>y</sub> and Pd<sub>x</sub>Nb<sub>1-x</sub>H<sub>y</sub> hydrides on CO<sub>2</sub> reduction. Changzhi Ai, Jin Hyun Chang, Alexander Sougaard Tygesen, Tejs Vegge and Heine Anton Hansen, _ChemSusChem_, **2024**, 17, e202301277.](https://doi.org/10.1002/cssc.202301277)

4. Graph neural network with multitasking genetic algorithm method was applied to screen Pd<sub>x</sub>Ti<sub>1-x</sub>H<sub>y</sub> with adsorbates under Various CO<sub>2</sub> Reduction Reaction Conditions. The code can be found in the folder [`instances/instance4_ml_ga`](https://github.com/changzhiai/pcat/tree/master/instances/instance4_ml_ga) and the details can be found in this paper:

    [Graph neural network-accelerated multitasking genetic algorithm for optimizing Pd<sub>x</sub>Ti<sub>1-x</sub>H<sub>y</sub> surfaces under various CO<sub>2</sub> reduction reaction conditions. Changzhi Ai, Shuang Han, Xin Yang, Tejs Vegge and Heine Anton Hansen, _ACS Appl. Mater. Interfaces_, **2024**, 16, 12563–12572.](https://doi.org/10.1021/acsami.3c18734)

## Contact
Changzhi Ai (changai@dtu.dk) at the Section of Atomic Scale Materials Modelling, Department of Energy Conversion and Storage, Technical University of Denmark.
