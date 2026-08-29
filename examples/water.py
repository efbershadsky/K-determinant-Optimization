#6-31G calculations for the water molecule
#In this usage example molecular integrals are calculated with PyQInt (https://github.com/ifilot/pyqint).
#import packages
import k_det
from pyqint import PyQInt, Molecule
#initialize molecule and basis set using PyQint
mol = Molecule()
integrator = PyQInt()
#bohr radius
a0=0.529177210544
#experimantal geometry
mol.add_atom('O', 0.0, 0.0, 0.0)
mol.add_atom('H', 0.7572/a0, 0.5865/a0, 0.0)
mol.add_atom('H', -0.7572/a0, 0.5865/a0, 0.0)
#6-31G basis set
cgfs, nuclei = mol.build_basis('p631')
#calculate molecular integrals using PyQint
S, T, V, eri = integrator.build_integrals_openmp(cgfs, nuclei)
#calculate energy of nuclear repulsion
E_nuc_rep=k_det.calculate_nuclear_repulsion(nuclei)
#input molecular parameters
#number of electrons
n_electrons=10
#spin multiplicity
multiplicity=1
#random seed
rseed=42
#number of determinants
k=2
#run energy optimisation procedure using k_det
E,psi,data=k_det.optimize_molecular_energy(n_electrons,multiplicity,k,S,T,V,eri,E_nuc_rep,rseed)
