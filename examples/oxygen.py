#6-31G* calculations for the oxygen atom.
#This usage example uses pre-computed molecular integrals from the folder "oxygen atom"
import k_det
import numpy as np
#random seed
rseed=42
#download data from the folder "oxygen atom"
T=np.load('oxigen_kin.npy')
V=np.load('oxigen_pot.npy')
S=np.load('oxigen_overlap.npy')
eri=np.load('oxigen_eri.npy')
#input molecular parameters
#number of electrons
n_electrons=8
#spin multiplicity
multiplicity=3
#number of determinants
k=2
E_nuc_rep=0
E,psi,data=k_det.optimize_molecular_energy(n_electrons,multiplicity,k,S,T,V,eri,E_nuc_rep,rseed)
