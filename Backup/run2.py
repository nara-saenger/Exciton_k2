import numpy as np
import scipy
import pandas as pd
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import inputs as inp
import lattice
import kmc
import loading as ld


# this is an example of how can you run a KMC job with this code
# regular lattice

RF=2
tau = 1
r0=1
n       = 1000
L = r0*np.array([n])
PBC     = [0]
LOAD_FIXED_KAPPA=False
FIXED_KAPPA = 0.476#2/3
save = False
cutoff = r0
krot_dip = (n**(len(L)))*(RF**6/tau)
nrounds = 500
dim  = len(L)
#detail: Only one syst object, so all dynamics here have the same lattice and orientational profile
syst = ld.System(save=save,RF=RF,tau=tau,r0=r0,cutoff=cutoff,L=L,LOAD_FIXED_KAPPA=LOAD_FIXED_KAPPA,FIXED_KAPPA=FIXED_KAPPA,n=n,PBC=PBC,krot_dip=krot_dip)
syst.make_lattice()
results = Parallel(n_jobs=10,verbose=10)(delayed(kmc.kmc_run_fluor)(syst) for n in range(nrounds))
out_dict = kmc.process_results(results,file_header="",dim=dim,RF=RF,tau=tau,r0=r0,FIXED_KAPPA=FIXED_KAPPA)

