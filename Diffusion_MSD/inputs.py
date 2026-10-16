import numpy as np

########################################
n       = 10
PBC     = [0,1,2]        #directions to impose PBC. x= 0, y=1, z = 2
RF      = 3.27 #forster radius WITHOUT kappa2
r0      = 1 #lattice parameter
L       = r0*np.array([n,n,n])#,n,n]#[1,1,1]      # box sizes in X,Y,Z
tau     = 100
cutoff  = r0
LOAD_FIXED_KAPPA = False#False
FIXED_KAPPA      = 2/3
krot_dip         = 0 #rate of dipole re-orientation
#####################################
BUILD_LATT       = True
BUILD_DIPO       = True
BUILD_RATES      = True
save             = True
#####################################
readlatt       = 'lattice.txt'
adjmat         = 'adjmat.txt'
adjmat_normvec = 'adjmatvec.npy'
readdipole     = 'dipoles.txt'
adjmat_kappa   = 'kappa_mat.txt'
rate_mat       = 'rates.txt'
adjmat_distvec = 'adjmat_distances_vec.npy'
prob_mat       = 'prob.txt'
ratesum_vec    = 'ratesum.txt'
#####################################
#print('DONE READING INPUT')
