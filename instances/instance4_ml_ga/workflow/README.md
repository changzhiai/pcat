# Active Learning Workflow with Multitasking Genetic Algorithm

This Autonomous materials discovery workflow integrates **active learning** with a **multitasking genetic algorithm (MTGA)** to accelerate the discovery and optimization of **complex catalytic surfaces** under **diverse reaction conditions**. The goal is to identify **optimal surface–multiple adsorbates configurations** — spanning **composition, geometry, and adsorption sites** — that deliver robust performance across **multiple reaction environments**.  

---

## 🧪 Example Reaction Conditions

In this workflow, **reaction conditions** are treated as **tasks** for the multitasking genetic algorithm (MTGA).  Each task is described by a set of thermodynamic and chemical potential parameters such as temperature, electrochemical potential, pH, and partial pressures of gas species.  

### Example Input Table

| d_mu_Pd | d_mu_Ti  | d_mu_H  |   T   |  U  | pH |   P_H2   |  P_CO2   |  P_H2O  |   P_CO   | kappa |
|---------|----------|---------|-------|-----|----|----------|----------|---------|----------|-------|
| -2.249  | -7.28453 | -3.6135 | 283.15| 0.0 |0.0 | 101325.0 | 101325.0 | 3534.0  | 0.101325 | 0.0   |
| -2.249  | -7.53453 | -3.6135 | 283.15| 0.0 |0.0 | 101325.0 | 101325.0 | 3534.0  | 0.101325 | 0.0   |
| -2.249  | -7.78453 | -3.6135 | 283.15| 0.0 |0.0 | 101325.0 | 101325.0 | 3534.0  | 0.101325 | 0.0   |
| -2.249  | -8.03453 | -3.6135 | 283.15| 0.0 |0.0 | 101325.0 | 101325.0 | 3534.0  | 0.101325 | 0.0   |
| -2.249  | -8.28453 | -3.6135 | 283.15| 0.0 |0.0 | 101325.0 | 101325.0 | 3534.0  | 0.101325 | 0.0   |

**Columns represent:**  
- `d_mu_Pd`, `d_mu_Ti`, `d_mu_H`: Chemical potential shifts of Pd, Ti, and H.  
- `T`: Temperature (K).  
- `U`: Applied electrochemical potential (V).  
- `pH`: Proton concentration descriptor.  
- `P_H2`, `P_CO2`, `P_H2O`, `P_CO`: Partial pressures of gas-phase species (Pa).  
- `kappa`: Screening parameter for electrochemical double layer.  

---

These parameters define **environment-specific tasks** for the MTGA, ensuring that surface–adsorbate structures are optimized **robustly across different operating conditions**.  

---


## 📊 Workflow Schematic

<img width="618" height="564" alt="workflow" src="https://github.com/user-attachments/assets/40596e9b-b16a-4bce-bf7c-d64704e71517" />

1. **Initialization**  
   - Construct candidate surface models decorated with **multiple adsorbates**.  
   - Define the configurational and environmental search space.  

2. **Surrogate Model Training**  
   - Train an **ensemble of equivariant graph neural networks (EGNNs)** to predict adsorption and reaction energetics.  
   - The ensemble provides both **predictions** and **uncertainty estimates**.  

3. **Multitasking Genetic Algorithm (MTGA)**  
   - Explore the configurational landscape using **mutation** and **crossover** operators.  
   - Transfer knowledge across **different reaction conditions** to enhance optimization efficiency.  

4. **Active Data Selection**  
   - Select candidates with **high model uncertainty** or **promising predicted performance**.  
   - Reduce redundant sampling and focus resources on informative structures.  

5. **First-Principles Calculations**  
   - Perform **DFT evaluations** of selected structures.  
   - Generate accurate reference data for retraining the surrogate model.  

6. **Iterative Refinement**  
   - Incorporate new DFT results into the dataset.  
   - Retrain the EGNN ensemble.  
   - Repeat MTGA search and active learning until convergence.  

---

## ✨ Key Features

- **Efficiency:** Minimizes costly DFT calculations through selective data acquisition.  
- **Global Exploration:** MTGA avoids local minima and explores broad configurational space.  
- **Multi-Condition Robustness:** Optimizes surface performance under varying environments.  
- **Uncertainty Awareness:** Actively targets high-value data to improve model accuracy.  

---

## 🚀 Applications

- Catalytic **surface structure optimization**  
- **Mixed-adsorbate** structure discovery  
- **Adsorption energy** and reactivity prediction  
- **Reaction pathway** screening under variable conditions  
- Autonomous **materials discovery workflows**  









