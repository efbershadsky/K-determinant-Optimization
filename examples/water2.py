#6-31G* calculations for the water molecule
#This usage example uses pre-computed molecular integrals from the folder "water molecule"
import k_det
import numpy as np
rseed=42
#download data from the folder "water molecule"
T=np.load('water_kin.npy')
V=np.load('water_pot.npy')
S=np.load('water_overlap.npy')
eri=np.load('water_eri.npy')
E_nuc_rep_arr=np.load('water_E_nuc_rep.npy')
E_nuc_rep=E_nuc_rep_arr[0]
#
n_electrons=10
multiplicity=1
k=2
E,psi,data=k_det.optimize_molecular_energy(n_electrons,multiplicity,k,S,T,V,eri,E_nuc_rep,rseed)
