import numpy as np
import math
import time
import scipy
import copy

def optimize_molecular_energy(n_electrons,multiplicity,k,overlap,kinetic,pot,eri,E_nuc,rseed):
    #this function performs k-determinant optimization of molecular energy
    #programm uses Hartree units
    #input data: 
    #number of electrons in the system: n_electrons
    #multiplicity of the system (2*S+1): Spin multiplicity
    #number of determinants in optimization procedure: k
    #overlap matrix: ovelap
    #kinetic energy matrix: kinetic
    #nuclear attraction matrix: pot
    #ERI tensor: eri
    #nuclear repusion energy: E_nuc
    #random seed: rseed
    #output data:
    #system energy (result of the optimization procedure): E
    #wave function: psi. triple array [i,j,k],where i iterates through basis vectors, j iterates through electrons, k iterates through determinants
    #array of output data: output data in the format: [computational time, number of iterations taken by BFGS solver, number of degrees of freedom, energy,gradient norm]
    time1=time.time()
    print("Welcome to k-determinant optimization of molecular energy!")
    #initialize random generator
    rng = np.random.default_rng(rseed)
    #total one-electron interactions matrix
    en=kinetic+pot     
    #size of basis set
    N_basis=len(overlap)
    #output input data
    print("random seed:",rseed)
    print("number of determinants:",k)
    print("number of basis functions:",N_basis)
    print("number of electrons:",n_electrons)
    #number of spin up and spin down electrons
    n_up=round((n_electrons+multiplicity-1)/2)
    n_down=round((n_electrons-multiplicity+1)/2)
    print("number of spin up electrons:", n_up)
    print("number of spin down electrons:",n_down)
    #this condition tests mutiplicity
    if not ((n_electrons-multiplicity)%2==1):
        print("Impossible multiplicity value.")
        exit()
    #this condition tests number of electrons
    if n_electrons<2:
        print("Impossible number of electrons. Method works only with systems with at least two electrons.")
        exit()
    if (n_up<0 or n_down<0):
        print("Impossible number of n_up or n_down electrons.")
        exit()
    print("Performing one-determinant optimization")
    V=random_initialization(N_basis,n_electrons,overlap,n_up,n_down,rng)
    time0=time.time()
    res=scipy.optimize.minimize(minimizer,V,jac=True,args=(overlap,eri,n_electrons,N_basis,en,1,n_up,n_down,E_nuc),method='BFGS',options={'gtol': 1e-8})
    #print(res)
    T_uhf=time.time()-time0
    T_est=round(T_uhf*pow(k,3))
    E=res.fun
    psi=np.reshape(res.x,(N_basis,n_electrons,1))
    print("Calculated 1-determinant (UHF) energy:",np.asarray([E]),'hartree')
    if k>1:
        print("Estimated computational time:",T_est,'seconds')
        print("Performing k-determinant optimization")
        print("Number of degrees of freedom:",N_basis*n_electrons*k)
        U1=np.reshape(res.x,(N_basis,n_electrons))
        U1=normalize_U(U1,overlap,n_electrons,n_up,n_down)
        V2=structured_initialization(overlap,n_electrons,N_basis,k,n_up,n_down,U1,rng)
        #optimization via BFGS. fast, O(D^2)=O((N_basis*n_electrons*k)^2) memory.
        res=scipy.optimize.minimize(minimizer,V2,jac=True,args=(overlap,eri,n_electrons,N_basis,en,k,n_up,n_down,E_nuc),method='BFGS',options={'gtol': 1e-6})
        #print(res)
        E=res.fun
        print ("Calculated k-determinant energy:",np.asarray([E]),'hartree')
        psi=np.reshape(res.x,(N_basis,n_electrons,k))    
    time2=time.time()    
    print("computational time:",time2-time1,'seconds') 
    jac_norm=np.sqrt(np.sum(res.jac*res.jac))/np.sqrt(len(res.jac))
    #output data in the format: [computational time, number of iterations taken by solver, number of degrees of freedom, energy,gradient norm]
    data=[time2-time1,res.nit,N_basis*n_electrons*k,np.asarray([E]),jac_norm]     
    return np.asarray([E]),psi,data

def random_initialization(N_basis,n_electrons,overlap,n_up,n_down,rng):
    V=rng.standard_normal((N_basis,n_electrons))
    V1=normalize_U(V,overlap,n_electrons,n_up,n_down)
    V1=np.reshape(V1,N_basis*n_electrons)
    return V1

def structured_initialization(overlap,n_electrons,N_basis,k,n_up,n_down,U1,rng):
    # basic start matrix
    V2=[]
    for s in range(k-1):
        V0=copy.deepcopy(U1)
        V0[:,0]=rng.standard_normal(N_basis)
        V0[:,1]=rng.standard_normal(N_basis)
        V0=normalize_U(V0,overlap,n_electrons,n_up,n_down)
        V0=V0*pow(0.01,1/(2*n_electrons))
        V1=np.reshape(V0,N_basis*n_electrons)
        if s==0:
            V2=V1
        else:
            V2=np.concatenate((V2,V1))    
    U2=np.reshape(U1,(N_basis*n_electrons))
    V2=np.concatenate((V2,U2))
    return V2

def normalize_U(U1,overlap,n_electrons,n_up,n_down):
    sc=calculate_state_determinant_spin(overlap,n_electrons,U1[:,0:n_up],U1[:,0:n_up],U1[:,n_up:n_electrons],U1[:,n_up:n_electrons],n_up,n_down)
    U2=U1/pow(sc,1/(2*n_electrons))
    return U2

def calculate_state_determinant_spin(overlap,n_electrons,U1,V1,U2,V2,n_up,n_down):
    import numpy as np
    #for spin up part
    C1=np.einsum('st,si,tj->ij',overlap,U1,V1, optimize='greedy')
    #C1=U1@overlap@V1
    #for spin down part
    C2=np.einsum('st,si,tj->ij',overlap,U2,V2, optimize='greedy')
    #C2=U2@overlap@V2
    return np.linalg.det(C1)*np.linalg.det(C2)

def get_basic_matrixes(U1,V1,U2,V2,en,overlap,n_up,n_down):
    #this function calculates overlap matrixes and their determinants
    #for spin up part
    C1=np.einsum('st,si,tj->ij',overlap,U1,V1, optimize='greedy')
    #for spin down part
    C2=np.einsum('st,si,tj->ij',overlap,U2,V2, optimize='greedy')
    #
    C3=np.einsum('st,si->it',overlap,U1, optimize='greedy')
    #
    C4=np.einsum('st,si->it',overlap,U2, optimize='greedy')
    #for spin up part
    B1=np.einsum('st,si,tj->ij',en,U1,V1, optimize='greedy')
    #for spin down part
    B2=np.einsum('st,si,tj->ij',en,U2,V2, optimize='greedy')
    #
    B3=np.einsum('st,si->it',en,U1, optimize='greedy')
    #
    B4=np.einsum('st,si->it',en,U2, optimize='greedy')
    return C1,C2,C3,C4,B1,B2,B3,B4

def get_matrixes(i,j,C01,C02,C03,C04,n_up,n_down):
    C1=np.zeros((n_up,n_up))
    C2=np.zeros((n_down,n_down))
    C1[:,:]=C01[:,:]
    C2[:,:]=C02[:,:]
    if i<n_up:        
        C1[:,i]=C03[:,j]
    else:
        i1=i-n_up
        C2[:,i1]=C04[:,j]
    return C1,C2

def get_basic_tensors(U,V,eri,str,n_up,n_down):
    time1=time.time()
    n_electrons=n_up+n_down
    U1=U[:,0:n_up]
    V1=V[:,0:n_up]
    U2=U[:,n_up:n_electrons]
    V2=V[:,n_up:n_electrons]
    if str=='up':
        Tu3=np.einsum('abcd,ai,ck->ibkd',eri,U1,U1, optimize='greedy')
        Tuu=np.einsum('ibkd,bj,dl->ikjl',Tu3,V1,V1, optimize='greedy')
        #Tuu=np.einsum('abcd,ai,ck,bj,dl->ikjl',eri,U1,U1,V1,V1, optimize='greedy')
        #Tu1=np.einsum('abcd,ai,ck,bj->ikjd',eri,U1,U1,V1, optimize='greedy')
        Tu1=np.einsum('ibkd,bj->ikjd',Tu3,V1, optimize='greedy')
        #Tu2=np.einsum('abcd,ai,ck,dl->ibkl',eri,U1,U1,V1, optimize='greedy')  
        Tu2=np.einsum('ibkd,dl->ibkl',Tu3,V1, optimize='greedy')       
    elif str=='down':
        Tu3=np.einsum('abcd,ai,ck->ibkd',eri,U2,U2, optimize='greedy')
        Tuu=np.einsum('ibkd,bj,dl->ikjl',Tu3,V2,V2, optimize='greedy')
        #Tuu=np.einsum('abcd,ai,ck,bj,dl->ikjl',eri,U2,U2,V2,V2, optimize='greedy')
        #Tu1=np.einsum('abcd,ai,ck,bj->ikjd',eri,U2,U2,V2, optimize='greedy')
        Tu1=np.einsum('ibkd,bj->ikjd',Tu3,V2, optimize='greedy')
        #Tu2=np.einsum('abcd,ai,ck,dl->ibkl',eri,U1,U1,V2, optimize='greedy')  
        Tu2=np.einsum('ibkd,dl->ibkl',Tu3,V2, optimize='greedy')        
    elif str=='mixed': 
        Tu3=np.einsum('abcd,ai,ck->ibkd',eri,U1,U2, optimize='greedy')
        Tuu=np.einsum('ibkd,bj,dl->ikjl',Tu3,V1,V2, optimize='greedy')
        #Tuu=np.einsum('abcd,ai,ck,bj,dl->ikjl',eri,U1,U2,V1,V2, optimize='greedy')
        #Tu1=np.einsum('abcd,ai,ck,bj->ikjd',eri,U1,U2,V1, optimize='greedy')
        Tu1=np.einsum('ibkd,bj->ikjd',Tu3,V1, optimize='greedy')
        #Tu2=np.einsum('abcd,ai,ck,dl->ibkl',eri,U1,U2,V2, optimize='greedy')
        Tu2=np.einsum('ibkd,dl->ibkl',Tu3,V2, optimize='greedy')
        Tu3=[]
    #print(time.time()-time1)
    return Tuu,Tu1,Tu2,Tu3

def get_eri_tensors(U,V,i1,k1,eri,n_up,n_down,N_basis,Tuu,Tu1,Tu2,Tu3,Tdd,Td1,Td2,Td3,Tud,Tud1,Tud2,Tud3):
    Tuu_new=copy.deepcopy(Tuu)
    Tdd_new=copy.deepcopy(Tdd)
    Tud_new=copy.deepcopy(Tud)
    if i1<n_up:        
        Tuu_new[:,:,i1,:]=Tu2[:,k1,:,:]
        Tuu_new[:,:,:,i1]=Tu1[:,:,:,k1]
        Tuu_new[:,:,i1,i1]=Tu3[:,k1,:,k1]
        Tud_new[:,:,i1,:]=Tud2[:,k1,:,:]
    if i1>=n_up:
        i1=i1-n_up        
        Tdd_new[:,:,i1,:]=Td2[:,k1,:,:]
        Tdd_new[:,:,:,i1]=Td1[:,:,:,k1]
        Tdd_new[:,:,i1,i1]=Td3[:,k1,:,k1]  
        Tud_new[:,:,:,i1]=Tud1[:,:,:,k1]   
    return Tuu_new,Tdd_new,Tud_new

def get_contracted_tensors(Tud,Tud1,Tud2,Q01,Q02,E1,E2):
    Y1=Q01*E1
    Y2=Q02*E2
    #np.einsum('ikjl,ij,kl->',Tud,Y1,Y2, optimize='greedy')
    Tud_cup=np.einsum('ikjl,kl->ij',Tud,Y2,optimize='greedy',dtype=np.float64)
    Tud_cup1=np.einsum('ibkl,kl->ib',Tud2,Y2,optimize='greedy',dtype=np.float64)
    Tud_cdown=np.einsum('ikjl,ij->kl',Tud,Y1,optimize='greedy',dtype=np.float64)
    Tud_cdown1=np.einsum('ikjd,ij->kd',Tud1,Y1,optimize='greedy',dtype=np.float64)
    return Tud_cup,Tud_cup1,Tud_cdown,Tud_cdown1

def process_contracted_tensors(Tud_cup,Tud_cup1,Tud_cdown,Tud_cdown1,n_up,n_down,i1,k1):
    if i1<n_up:
        Tudc=copy.deepcopy(Tud_cup)    
        Tudc[:,i1]=Tud_cup1[:,k1]
    else:
        i2=i1-n_up
        Tudc=copy.deepcopy(Tud_cdown)
        Tudc[:,i2]=Tud_cdown1[:,k1] 
    return Tudc

def minimizer(V2,overlap,eri,n_electrons,N_basis,en,k,n_up,n_down,E_nuc):
    #time1=time.time()
    #this function calculates energy of a given k-determinant state and its gradient
    Q1=np.zeros(k*N_basis*n_electrons)
    Q2=np.zeros(k*N_basis*n_electrons)
    Q=np.zeros(k*N_basis*n_electrons)
    E_full=0
    sc_full=0
    Tu3s=[]
    F01=prepare_sign_matrixes(n_up)
    F02=prepare_sign_matrixes(n_down)
    G01=prepare_sign_matrixes2(n_up)
    G02=prepare_sign_matrixes2(n_down)
    E_fulls=[]
    sc_fulls=[]
    for g in range(k):
        A0=np.zeros((N_basis,n_electrons,k))
        A1=np.zeros((N_basis,n_electrons,k))
        for s in range(k):        
            V30=V2[s*N_basis*n_electrons:(s+1)*N_basis*n_electrons]
            V30=np.reshape(V30,(N_basis,n_electrons))
            V3=copy.deepcopy(V30)
            V40=V2[g*N_basis*n_electrons:(g+1)*N_basis*n_electrons]
            V40=np.reshape(V40,(N_basis,n_electrons))
            V4=copy.deepcopy(V40)
            Tuu,Tu1,Tu2,Tu3=get_basic_tensors(V3,V4,eri,'up',n_up,n_down)
            Tdd,Td1,Td2,Td3=get_basic_tensors(V3,V4,eri,'down',n_up,n_down)
            Tud,Tud1,Tud2,Tud3=get_basic_tensors(V3,V4,eri,'mixed',n_up,n_down)
            C01,C02,C03,C04,B01,B02,B03,B04=get_basic_matrixes(V3[:,0:n_up],V4[:,0:n_up],V3[:,n_up:n_electrons],V4[:,n_up:n_electrons],en,overlap,n_up,n_down)           
            E_arr,En_up0,En_down0=calculate_state_energy_spin2(overlap,eri,n_electrons,en,V3[:,0:n_up],V4[:,0:n_up],V3[:,n_up:n_electrons],V4[:,n_up:n_electrons],n_up,n_down,Tuu,Tdd,Tud,G01,G02)
            sc=np.linalg.det(C01)*np.linalg.det(C02)
            #E_full=E_full+E
            #sc_full=sc_full+sc
            #list energy components in array
            E_fulls.append(E_arr[0])
            E_fulls.append(E_arr[1])
            E_fulls.append(E_arr[2])
            E_fulls.append(E_arr[3])
            sc_fulls.append(sc)
            Q01=prepare_minors(C01,n_up)
            Q02=prepare_minors(C02,n_down)
            Tud_cup,Tud_cup1,Tud_cdown,Tud_cdown1=get_contracted_tensors(Tud,Tud1,Tud2,Q01,Q02,F01,F02)
            En1el_up0=calculate_one_electron_energy_component(n_up,C01,B01)
            En1el_down0=calculate_one_electron_energy_component(n_down,C02,B02)
            for i in range(n_electrons):
                for j in range(N_basis):
                    vec=np.zeros(N_basis)
                    V5=np.zeros((N_basis,n_electrons))
                    vec[j]=1
                    V5=copy.deepcopy(V4)
                    V5[:,i]=vec
                    Tuu_new,Tdd_new,Tud_new=get_eri_tensors(V3,V4,i,j,eri,n_up,n_down,N_basis,Tuu,Tu1,Tu2,Tu3,Tdd,Td1,Td2,Td3,Tud,Tud1,Tud2,Tud3)
                    C1,C2=get_matrixes(i,j,C01,C02,C03,C04,n_up,n_down)
                    B1,B2=get_matrixes(i,j,B01,B02,B03,B04,n_up,n_down)
                    Tudc=process_contracted_tensors(Tud_cup,Tud_cup1,Tud_cdown,Tud_cdown1,n_up,n_down,i,j)
                    #E grad
                    q1=calculate_state_energy_spin_tensors(overlap,eri,n_electrons,en,V3[:,0:n_up],V5[:,0:n_up],V3[:,n_up:n_electrons],V5[:,n_up:n_electrons],n_up,n_down,Tuu_new,Tdd_new,Tud_new,C1,C2,B1,B2,En_up0,En_down0,i,Q01,Q02,F01,F02,G01,G02,Tudc,En1el_up0,En1el_down0)                    
                    #det grad              
                    q2=np.linalg.det(C1)*np.linalg.det(C2)
                    #A[j,i]=A[j,i]+2*(q1/sc0)-2*E0*q2/(sc0*sc0)
                    A0[j,i,s]=q1
                    A1[j,i,s]=q2
        #A01tc=np.sum(A0,axis=2)
        #A02tc=np.sum(A1,axis=2)
        A01t=process_axis_summation(A0,N_basis,n_electrons,k)
        A02t=process_axis_summation(A1,N_basis,n_electrons,k)
        A02=np.reshape(A01t,N_basis*n_electrons)
        A12=np.reshape(A02t,N_basis*n_electrons)
        Q1[g*n_electrons*N_basis:(g+1)*N_basis*n_electrons]=A02[:]
        Q2[g*n_electrons*N_basis:(g+1)*N_basis*n_electrons]=A12[:]
    E_full=math.fsum(E_fulls)
    sc_full=math.fsum(sc_fulls)
    Q=Q1*2/sc_full-2*E_full*Q2/(sc_full*sc_full)
    E_state=E_full/sc_full
    grad=Q
    return E_state+E_nuc, grad

def process_axis_summation(A,N_basis,n_electrons,k):
    B=np.zeros((N_basis,n_electrons),dtype=np.float64)
    for i in range(n_electrons):
        for j in range(N_basis):
            c=A[j,i,:]
            B[j,i]=math.fsum(c)
    return B

def calculate_one_electron_energy(n_up,n_down,C1,C2,M1,M2,B1,B2):
    #1-electron energy
    En1_up=calculate_one_electron_energy_component(n_up,C1,B1)
    En1_down=calculate_one_electron_energy_component(n_down,C2,B2)
    En=M2*En1_up+M1*En1_down
    return En

def calculate_one_electron_energy_component(n_up,C1,B1):
    Q1=np.zeros((n_up,n_up,n_up))
    F1=np.zeros((n_up,n_up)) 
    for i in range(n_up):
        F1[:,:]=C1[:,:]
        F1[:,i]=B1[:,i]
        Q1[i,:,:]=F1[:,:]
    Qup=np.linalg.det(Q1)
    En1=math.fsum(Qup)
    return En1

def calculate_overlap_matrixes(overlap,n_electrons,en,U1,V1,U2,V2,n_up,n_down):
    #this function calculates overlap matrixes and their determinants
    #for spin up part
    C1=np.einsum('st,si,tj->ij',overlap,U1,V1, optimize='greedy')
    #for spin down part
    C2=np.einsum('st,si,tj->ij',overlap,U2,V2, optimize='greedy')
    M1=np.linalg.det(C1)
    M2=np.linalg.det(C2)
    return C1,C2,M1,M2

def calculate_en_matrixes(overlap,n_electrons,en,U1,V1,U2,V2,n_up,n_down):
    B1=np.einsum('st,si,tj->ij',en,U1,V1, optimize='greedy')
    B2=np.einsum('st,si,tj->ij',en,U2,V2, optimize='greedy')
    return B1,B2

def calculate_tensors_directly2(T1,T2,T3):
    #this function calculates two-electron orbital integrals directly from eri tensor
    #
    #T01=np.einsum('abcd,ai,ck,bj,dl->ikjl',eri,U1,U1,V1,V1, optimize='greedy')
    T01=T1
    T02=np.transpose(T01,(1,0,2,3))
    Tuu=T01-T02
    #
    #T01=np.einsum('abcd,ai,ck,bj,dl->ikjl',eri,U2,U2,V2,V2, optimize='greedy')
    T01=T2
    T02=np.transpose(T01,(1,0,2,3))
    Tdd=T01-T02
    #
    Tud=T3    
    return Tuu,Tdd,Tud

def calculate_two_electron_energy_same_spin(C1,M2,Tuu,n_up):
    En=0
    if n_up>0:
        D1=np.zeros((n_up,n_up,n_up,n_up-1,n_up-1))
        E1=np.zeros((n_up,n_up,n_up))
        for i in range(n_up):
            for k in range(i):
                for j in range(n_up):                   
                    W10=np.delete(C1,i,axis=0)
                    W1=np.delete(W10,j,axis=1) 
                    for l in range(n_up-1):
                        if l<j:
                            W1[k,l]=Tuu[i,k,j,l]
                        else:
                            W1[k,l]=Tuu[i,k,j,l+1]
                    #En4=En4+0.5*pow(-1,i+j)*np.linalg.det(B1)
                    D1[i,j,k,:,:]=W1[:,:]
                    E1[i,j,k]=0.5*pow(-1,i+j)
        Q1=np.linalg.det(D1)
        En=En+M2*np.sum(Q1*E1)
    return En

def calculate_two_electron_energy_same_spin_fast(C1,Tuu,n_up,E1):
    En=0
    if n_up>0:
        D1=np.zeros((n_up,n_up,n_up,n_up-1,n_up-1))
        W11=np.zeros((n_up-1,n_up-1))
        for j in range(n_up):
            for i in range(n_up):
                W11[0:i,0:j]=C1[0:i,0:j]
                W11[0:i,j:n_up-1]=C1[0:i,j+1:n_up]
                W11[i:n_up-1,0:j]=C1[i+1:n_up,0:j]
                W11[i:n_up-1,j:n_up-1]=C1[i+1:n_up,j+1:n_up]
                D1[i,j,0:i,:,:]= W11[:,:]            
                for k in range(i):    
                    D1[i,j,k,k,0:j]=Tuu[i,k,j,0:j]
                    D1[i,j,k,k,j:n_up-1]=Tuu[i,k,j,j+1:n_up]
        Q1=np.linalg.det(D1)
        P1=Q1*E1
        P2=np.reshape(P1,(n_up*n_up*n_up))
        En=math.fsum(P2)
        #En1=np.sum(Q1*E1)
    return En

def calculate_two_electron_energy_different_spin(C1,C2,Tud,n_up,n_down):
    En=0
    if (n_up>0 and n_down>0): 
        Y1=np.zeros((n_up,n_up))
        Y2=np.zeros((n_down,n_down))
        D1=np.zeros((n_up,n_up,n_up-1,n_up-1))
        E1=np.zeros((n_up,n_up))
        for i in range(n_up):
            for j in range(n_up):
                W10=np.delete(C1,i,axis=0)
                W1=np.delete(W10,j,axis=1) 
                #Y1[i,j]=np.linalg.det(W1)*pow(-1,i+j)
                D1[i,j,:,:]=W1[:,:]
                E1[i,j]=pow(-1,i+j)
        Y1=np.linalg.det(D1)*E1
        D2=np.zeros((n_down,n_down,n_down-1,n_down-1))
        E2=np.zeros((n_down,n_down))
        for k in range(n_down):
            for l in range(n_down):
                W10=np.delete(C2,k,axis=0)
                W2=np.delete(W10,l,axis=1) 
                #Y2[k,l]=np.linalg.det(W2)*pow(-1,k+l)
                D2[k,l,:,:]=W2[:,:]
                E2[k,l]=pow(-1,k+l)
        Y2=np.linalg.det(D2)*E2
        En=np.einsum('ikjl,ij,kl->',Tud,Y1,Y2, optimize='greedy')
    return En

def calculate_two_electron_energy_different_spin_fast(C1,C2,Tud,n_up,n_down):
    En=0
    if (n_up>0 and n_down>0): 
        Q1=prepare_minors(C1,n_up)
        E1=prepare_sign_matrixes(n_up)
        Q2=prepare_minors(C2,n_down)
        E2=prepare_sign_matrixes(n_down)
        Y1=Q1*E1
        Y2=Q2*E2
        En=np.einsum('ikjl,ij,kl->',Tud,Y1,Y2, optimize='greedy')
    return En

def calculate_two_electron_energy_different_spin_fast_tensors(C1,C2,n_up,n_down,i,E1,E2,Tudc):
    En=0
    if (n_up>0 and n_down>0): 
        if i<n_up:
            Q1=prepare_minors(C1,n_up)
            #Q1=matrix_minors_svd(C1,E1,n_up)
            Y1=Q1*E1
            #En=np.einsum('ij,ij->',Tudc,Y1, optimize='greedy')
            P1=Tudc*Y1
            P2=np.reshape(P1,(n_up*n_up))
            En=math.fsum(P2)
            #En1=np.sum(Tudc*Y1)
        else:
            Q2=prepare_minors(C2,n_down)
            #Q2=matrix_minors_svd(C2,E2,n_down)
            Y2=Q2*E2
            P1=Tudc*Y2
            P2=np.reshape(P1,(n_down*n_down))
            En=math.fsum(P2)
            #En1=np.sum(Tudc*Y2)
            #En=np.einsum('ij,ij->',Tudc,Y2, optimize='greedy')
    return En


def calculate_two_electron_energy_different_spin_fast_tensors2(C1,C2,n_up,n_down,i,E1,E2,Tudc):
    En=0
    if (n_up>0 and n_down>0): 
        if i<n_up:
            F2=np.zeros((n_up,n_up,n_up))
            Ens=np.zeros(n_up)
            for i in range(n_up):
                F2[i,:,:]=C1[:,:]
                F2[i,:,i]=Tudc[:,i]
            Ens=np.linalg.det(F2)
            En=math.fsum(Ens)
        else:
            F2=np.zeros((n_down,n_down,n_down))
            Ens=np.zeros(n_down)
            for i in range(n_down):
                F2[i,:,:]=C2[:,:]
                F2[i,:,i]=Tudc[:,i]
            Ens=np.linalg.det(F2)
            En=math.fsum(Ens)
    return En

def prepare_minors(C1,n_up):
    if n_up>0:
        D1=np.zeros((n_up,n_up,n_up-1,n_up-1))
        W1=np.zeros((n_up-1,n_up-1))
        for i in range(n_up):
            for j in range(n_up):
                W1[0:i,0:j]=C1[0:i,0:j]
                W1[0:i,j:n_up-1]=C1[0:i,j+1:n_up]
                W1[i:n_up-1,0:j]=C1[i+1:n_up,0:j]
                W1[i:n_up-1,j:n_up-1]=C1[i+1:n_up,j+1:n_up]
                D1[i,j,:,:]=W1[:,:]
        Q1=np.linalg.det(D1)
    else:
        Q1=prepare_sign_matrixes(n_up)
    return Q1

def prepare_sign_matrixes(n_up):
    E1=np.zeros((n_up,n_up))
    for i in range(n_up):
        for j in range(n_up):
            E1[i,j]=pow(-1,i+j)
    return E1

def prepare_sign_matrixes2(n_up):
    E1=np.zeros((n_up,n_up,n_up))
    for j in range(n_up):
        for i in range(n_up):
            E1[i,j,0:i]=0.5*pow(-1,i+j) 
    return E1

def calculate_state_energy_spin2(overlap,eri,n_electrons,en,U1,V1,U2,V2,n_up,n_down,T1,T2,T3,G01,G02):
    E_arr=[]
    #1-electron energy
    C1,C2,M1,M2=calculate_overlap_matrixes(overlap,n_electrons,en,U1,V1,U2,V2,n_up,n_down)
    B1,B2=calculate_en_matrixes(overlap,n_electrons,en,U1,V1,U2,V2,n_up,n_down)
    En1=calculate_one_electron_energy(n_up,n_down,C1,C2,M1,M2,B1,B2)
    E_arr.append(En1)
    #print(En1)
    #2-electron energy
    #calculate two-electron orbital integrals directly from eri tensor
    Tuu,Tdd,Tud=calculate_tensors_directly2(T1,T2,T3)
    #component related to interaction between spin up and spin up electrons
    En_up=calculate_two_electron_energy_same_spin_fast(C1,Tuu,n_up,G01)
    E_arr.append(En_up*M2)
    #component related to interaction between spin down and spin down electrons
    En_down=calculate_two_electron_energy_same_spin_fast(C2,Tdd,n_down,G02)
    E_arr.append(En_down*M1)
    #component related to interaction between spin up and spin down electrons
    En_mixed=calculate_two_electron_energy_different_spin_fast(C1,C2,Tud,n_up,n_down)
    E_arr.append(En_mixed)
    #En2=En_up*M2+En_down*M1+En_mixed
    #En=math.fsum(E_arr)
    return E_arr,En_up,En_down

def calculate_tensors_from_prototypes(eri,n_electrons,U1,V1,U2,V2,n_up,n_down,Tuu0,Tdd0,Tud):
    #T01=np.einsum('abcd,ai,ck,bj,dl->ikjl',eri,U1,U1,V1,V1, optimize='greedy')
    T01=copy.deepcopy(Tuu0)
    T02=np.transpose(T01,(1,0,2,3))
    Tuu=T01-T02
    #
    #T01=np.einsum('abcd,ai,ck,bj,dl->ikjl',eri,U2,U2,V2,V2, optimize='greedy')
    T01=copy.deepcopy(Tdd0)
    T02=np.transpose(T01,(1,0,2,3))
    Tdd=T01-T02
    return Tuu,Tdd,Tud

def calculate_state_energy_spin_tensors(overlap,eri,n_electrons,en,U1,V1,U2,V2,n_up,n_down,Tuu0,Tdd0,Tud,C1,C2,B1,B2,En_up0,En_down0,i,Q01,Q02,F01,F02,G01,G02,Tudc,En1el_up0,En1el_down0):
    #1-electron energy
    M1=np.linalg.det(C1)
    M2=np.linalg.det(C2)
    #En1=calculate_one_electron_energy(n_up,n_down,C1,C2,M1,M2,B1,B2)
    #print(En1)
    if i<n_up:
        En1el_up=calculate_one_electron_energy_component(n_up,C1,B1)
    else:
        En1el_up=En1el_up0
    if i>=n_up:
        En1el_down=calculate_one_electron_energy_component(n_down,C2,B2)
    else:
        En1el_down=En1el_down0  
    En1=M2*En1el_up+M1*En1el_down
    #print(En1)
    #2-electron energy
    #calculate two-electron orbital integrals using pre-computed state-related data
    Tuu,Tdd,Tud=calculate_tensors_from_prototypes(eri,n_electrons,U1,V1,U2,V2,n_up,n_down,Tuu0,Tdd0,Tud)
    #component related to interaction between spin up and spin up electrons
    if i<n_up: 
        En_up=calculate_two_electron_energy_same_spin_fast(C1,Tuu,n_up,G01)
    else:
        En_up=En_up0
    #component related to interaction between spin down and spin down electrons
    if i>=n_up: 
        En_down=calculate_two_electron_energy_same_spin_fast(C2,Tdd,n_down,G02)
    else:
        En_down=En_down0
    #component related to interaction between spin up and spin down electrons
    #En_mixed=calculate_two_electron_energy_different_spin_fast(C1,C2,Tud,n_up,n_down)
    En_mixed=calculate_two_electron_energy_different_spin_fast_tensors(C1,C2,n_up,n_down,i,F01,F02,Tudc)
    #En_mixed=calculate_two_electron_energy_different_spin_fast_tensors2(C1,C2,n_up,n_down,i,F01,F02,Tudc)
    En2=En_up*M2+En_down*M1+En_mixed
    return En1+En2

def calculate_nuclear_repulsion(nuclei):
    E=0
    for i in range(len(nuclei)):
        for j in range(i):
            r1=nuclei[i][0]
            r2=nuclei[j][0]
            q1=nuclei[i][1]
            q2=nuclei[j][1]
            r=r1-r2
            dist=np.sqrt(np.sum(r*r))
            E=E+q1*q2/dist
    return E

def calculate_all_products(psi,overlap,eri,n_electrons,N_basis,en,k,n_up,n_down,E_nuc):
    #time1=time.time()
    #this function calculates matrix of products H=<F{i}|H|F{j}> and R= <F{i}|F{j}> for set of determinants F{i} from wave function psi
    V2=np.reshape(psi,(N_basis*n_electrons*k))  
    Q1=np.zeros(k*N_basis*n_electrons)
    Q2=np.zeros(k*N_basis*n_electrons)
    Q=np.zeros(k*N_basis*n_electrons)
    H=np.zeros((k,k))
    R=np.zeros((k,k))
    E_full=0
    sc_full=0
    Tu3s=[]
    F01=prepare_sign_matrixes(n_up)
    F02=prepare_sign_matrixes(n_down)
    G01=prepare_sign_matrixes2(n_up)
    G02=prepare_sign_matrixes2(n_down)
    E_fulls=[]
    sc_fulls=[]
    for g in range(k):
        A0=np.zeros((N_basis,n_electrons,k))
        A1=np.zeros((N_basis,n_electrons,k))
        for s in range(k):        
            V30=V2[s*N_basis*n_electrons:(s+1)*N_basis*n_electrons]
            V30=np.reshape(V30,(N_basis,n_electrons))
            V3=copy.deepcopy(V30)
            V40=V2[g*N_basis*n_electrons:(g+1)*N_basis*n_electrons]
            V40=np.reshape(V40,(N_basis,n_electrons))
            V4=copy.deepcopy(V40)
            Tuu,Tu1,Tu2,Tu3=get_basic_tensors(V3,V4,eri,'up',n_up,n_down)
            Tdd,Td1,Td2,Td3=get_basic_tensors(V3,V4,eri,'down',n_up,n_down)
            Tud,Tud1,Tud2,Tud3=get_basic_tensors(V3,V4,eri,'mixed',n_up,n_down)
            C01,C02,C03,C04,B01,B02,B03,B04=get_basic_matrixes(V3[:,0:n_up],V4[:,0:n_up],V3[:,n_up:n_electrons],V4[:,n_up:n_electrons],en,overlap,n_up,n_down)           
            E_arr,En_up0,En_down0=calculate_state_energy_spin2(overlap,eri,n_electrons,en,V3[:,0:n_up],V4[:,0:n_up],V3[:,n_up:n_electrons],V4[:,n_up:n_electrons],n_up,n_down,Tuu,Tdd,Tud,G01,G02)
            sc=np.linalg.det(C01)*np.linalg.det(C02)
            #E_full=E_full+E
            #sc_full=sc_full+sc
            #list energy components in array
            E_fulls.append(E_arr[0])
            E_fulls.append(E_arr[1])
            E_fulls.append(E_arr[2])
            E_fulls.append(E_arr[3])
            sc_fulls.append(sc)
            H[g,s]=math.fsum(E_arr)
            R[g,s]=sc
    return H,R
