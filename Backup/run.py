import numpy as np
import scipy
import pandas as pd
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import lattice
import kmc
import loading as ld


# this is an example of how can you run a KMC job with this code
# regular lattice

RF=1
tau = 1
r0=1
n=10
L =  r0*np.array([n,n,n])
PBC = [0,1,2]
LOAD_FIXED_KAPPA= False
FIXED_KAPPA = 0.476#2/3
save = False
cutoff = r0
nrounds = 1000
krot_dip = 0 #=0 -> no rotation

#detail: Only one syst object, so all dynamics here have the same lattice and orientational profile
syst = ld.System(save=save,RF=RF,tau=tau,r0=r0,cutoff=cutoff,L=L,LOAD_FIXED_KAPPA=LOAD_FIXED_KAPPA,FIXED_KAPPA=FIXED_KAPPA,n=n,PBC=PBC,krot_dip=krot_dip)
syst.make_lattice()
results = Parallel(n_jobs=10,verbose=10)(delayed(kmc.kmc_run_fluor)(syst) for n in range(nrounds))
out_dict = kmc.process_results(results,file_header="",dim=len(L),RF=RF,tau=tau,r0=r0,FIXED_KAPPA=FIXED_KAPPA)

