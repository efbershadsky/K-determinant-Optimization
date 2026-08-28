#In this usage example molecular integrals are calculated with PyQInt (https://github.com/ifilot/pyqint).
#import packages
import k_det
from pyqint import PyQInt, Molecule
#initialize molecule and basis set using PyQint
mol = Molecule()
integrator = PyQInt()
mol.add_atom('C', 0.0, 0.0, 0.0)
#6-31G basis set
cgfs, nuclei = mol.build_basis('p631')
#calculate integrals using PyQint
S, T, V, eri = integrator.build_integrals_openmp(cgfs, nuclei)
#calculate energy of nuclear repulsion
E_nuc_rep=k_det.calculate_nuclear_repulsion(nuclei)
#input molecular parameters
#number of electrons
n_electrons=6
#spin multiplicity
multiplicity=3
#random seed
rseed=42
#number of determinants
k=2
#run energy optimisation procedure using q_phys
E,psi,data=k_det.optimize_molecular_energy(n_electrons,multiplicity,k,S,T,V,eri,E_nuc_rep,rseed)
