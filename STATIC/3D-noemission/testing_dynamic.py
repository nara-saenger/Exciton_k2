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
# - random dynamic dipoles
# - should return effective kappa of 2/3
##########################################
tau = 1
r0=1
RF = 3
n       = 10
L = r0*np.array([n])
PBC     = [0]
LOAD_FIXED_KAPPA  =False
FIXED_KAPPA = 2/3 # to compute 
REFERENCE_KAPPA =  2/3 #for the analysis
save = False
cutoff = r0
krot_dip = (n**(len(L)))*(20/tau)
nrounds = 100

krot_multiplicator = [0, 0.1, 10, 100, 130]
Ds,Lds = [],[]
for i,job in enumerate(krot_multiplicator):
    krot_dip = krot_multiplicator[i]*(RF**6/tau)
    syst = ld.System(save=save,krot_dip=krot_dip,RF=RF,tau=tau,r0=r0,cutoff=cutoff,L=L,LOAD_FIXED_KAPPA=LOAD_FIXED_KAPPA,FIXED_KAPPA=FIXED_KAPPA,PBC=PBC,n=n)
    syst.make_lattice()
    results = Parallel(n_jobs=20,verbose=10)(delayed(kmc.kmc_run_fluor)(syst) for n in range(nrounds))
    out_dict = kmc.process_results(results,file_header="",dim=len(L),RF=RF,tau=tau,r0=r0,FIXED_KAPPA=REFERENCE_KAPPA)
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
plt.savefig('testing_dynamic.png')
