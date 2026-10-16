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



#########################################
# Testing analytic LD in regular lattices
# - random fixed dipoles
# - should return effective kappa of 0.476
##########################################
tau = 1
r0=1
n=1000
L = r0*np.array([n])
LOAD_FIXED_KAPPA=False
FIXED_KAPPA = 0.476 #2/3
save = False
cutoff = r0
krot_dip = 0
PBC = [0]
nrounds = 100#00

jobs = [7,9,10]#[1,2,3,4,5]
Ds,Lds = [],[]
for job in jobs:
    rf = job
    syst = ld.System(save=save,RF=rf,tau=tau,r0=r0,cutoff=cutoff,L=L,LOAD_FIXED_KAPPA=LOAD_FIXED_KAPPA,FIXED_KAPPA=FIXED_KAPPA,n=n,PBC=PBC,krot_dip=krot_dip)
    syst.make_lattice()
    results = Parallel(n_jobs=10,verbose=10)(delayed(kmc.kmc_run_fluor)(syst) for _ in range(nrounds))
    out_dict = kmc.process_results(results,dim=len(L),RF=rf,tau=tau,r0=r0,FIXED_KAPPA=FIXED_KAPPA)
    Ds.append(out_dict['D'])
    Lds.append(out_dict['LD'])
    



Ds = np.array(Ds)
Lds = np.array(Lds)
fig,axs=plt.subplots(1,2)
axss =axs.flatten()
axss[0].scatter(Ds[:,0],Ds[:,1])
axss[1].scatter(Lds[:,0],Lds[:,1])


dmin = min(Ds[:,0].min(),Ds[:,1].min())
dmax = max(Ds[:,0].max(),Ds[:,1].max())
axss[0].set_xlim(dmin,dmax)
axss[0].set_ylim(dmin,dmax)
ldmin = min(Lds[:,0].min(),Lds[:,1].min())
ldmax = max(Lds[:,0].max(),Lds[:,1].max())
#axss[1].set_xlim(ldmin,ldmax)
#axss[1].set_ylim(ldmin,ldmax)
axss[0].set_xlabel('D analytic')
axss[0].set_ylabel('D calc')
axss[1].set_xlabel('LD analytic')
axss[1].set_ylabel('LD calc')
axss[0].plot([0, dmax], [0, dmax], '--', label = 'x=y',c='r')
axss[1].plot([0, ldmax], [0, ldmax], '--', label = 'x=y',c='r')
for ax in axss:
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    #ax.set_box_aspect(1)
plt.tight_layout()
plt.savefig('testing_static.png')
