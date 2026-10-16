#################################################
# KMC for random dipoles and dynamical sampling
# set - 2026
# Authors: Nara Saenger, Tiago Cassiano
################################################
import numpy as np
import scipy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import lattice
from copy import copy
reload_loading = True
if reload_loading:
  import loading as ld



def process_results(results,dim,RF,tau,r0,FIXED_KAPPA,file_header=""):

  Ts     = np.mean( [run[0] for run in results],axis=0)
  MSDs   = np.mean( [run[1] for run in results],axis=0)
  MSD_f  = np.mean( [run[2] for run in results],axis=0)
  D_c    = np.mean( [run[3] for run in results],axis=0)/dim
  D      = [ MSDs[i]/(2*dim*Ts[i]) for i in range(len(MSDs)) ]

  D_c_lis    =  np.array([run[3] for run in results])/dim
  kappa_D_lis = np.sqrt(D_c_lis*tau)*((r0)**2)/(RF)**3
  kappa2_D_lis = kappa_D_lis**2


  kappa2          = FIXED_KAPPA
  print(f'kappa2 fixo {kappa2}')
  LD_calc         = np.sqrt(MSD_f)/np.sqrt(2*dim)
  LD_ANALYTIC_1D  = (((kappa2)**(1/2))*(RF)**3) /(r0**2)
  D_ANALYTIC_1D   = (1/tau)*(((kappa2)*(RF)**6)/(r0**4))
  kappa_LD = LD_calc*((r0)**2)/(RF)**3
  kappa2_LD = kappa_LD**2

  MSD_lis =  np.array([run[2] for run in results])
  L_D_calc_lis = np.sqrt(MSD_lis)/np.sqrt(2*dim)
  kappa_LD_lis = L_D_calc_lis*((r0)**2)/(RF)**3
  kappa2_LD_lis = kappa_LD_lis**2

  print(f'ḱappa2 do LD {kappa2_LD}')

  T_int   = np.linspace(Ts.min(), Ts.max(), 500)
  MSD_int = np.interp(T_int, Ts, MSDs)
  np.savetxt(file_header+"MSD.txt",np.column_stack((T_int,MSD_int)))

  #np.savetxt(file_header+"Time.txt",T_int) 
   
  print('Dimensão',dim)
  print('Coeficientes de Difusão',D_ANALYTIC_1D,D[-1],D_c)
  print('LDs',LD_ANALYTIC_1D,LD_calc,np.sqrt(D_c*tau))
  out_dic = {'D':[D_ANALYTIC_1D,D_c[0]],'LD':[LD_ANALYTIC_1D,np.sqrt(D_c[0]*tau)], 'kappa2_LD': kappa2_LD, 'kappa_LD_lis': kappa2_LD_lis, 'kappa2_D_lis': kappa2_D_lis }
  return out_dic
  # D eff = 3 D_1D, where D_1D = MSD/2t <-- the '2' is already here


def kmc_run_fluor(syst):
  # making copies of thing that may change during dynamics
  # to avoid conflict with multiple independent runs in joblib
  local               = copy(syst)
  local.dipoles       = syst.dipoles.copy()
  local.kappa_adj_mat = syst.kappa_adj_mat.copy()
  local.rate_adj_mat  = syst.rate_adj_mat.copy()
  local.prob_adj_mat  = syst.prob_adj_mat.copy()
  local.ratesum       = syst.ratesum.copy()
  local.prob_dirty    = syst.prob_dirty.copy()


  rates         = local.rate_adj_mat
  probs         = local.prob_adj_mat
  ratesums      = local.ratesum
  distvec       = local.Adj_mat_distvec
  #initialization
  Nsites         = local.Nsites
  i              = int(Nsites*np.random.uniform(0,1))
  position       = np.array([0,0,0])
  time           = 0 
  position_fluor = np.array([0,0,0])
  time_fluor     = 0
  
  
  
  MSD         = [0]
  T           = [0]
  MSD_f          = []
  D           = []
  fluor_pos   = []
  fluor_times = []
  position_lis = [position]
  ALIVE       = True

  #for step in range(nsteps_max):
  while (ALIVE):
    if local.prob_dirty[i]: #if this particular site was involved in a previous dipole reorientation
        ld.update_prob_row(i,rates,probs,ratesums)
        local.prob_dirty[i] = False #now the prob is correct. Prob is nt dirty anymore
    probs_i = probs[i,:]
    ratesum = ratesums[i]
    
    u = np.random.uniform(0,1)
    event = np.searchsorted(probs_i, u, side="right")
    
    shift    = distvec[i,event]
    position = position + shift 
    position_fluor = position_fluor + shift
  
    
    dT         = -np.log(np.random.uniform(0,1))/ratesum
    time       = time  + dT
    time_fluor = time_fluor + dT

    position_lis.append(position)
    MSD.append(np.dot(position,position))
    T.append(time) 

    if i == event: #fluor
      MSD_f_i = np.dot(position_fluor,position_fluor)
      MSD_f.append(MSD_f_i)
      D.append(MSD_f_i/(2*time_fluor))
      fluor_pos.append(i)
      fluor_times.append(time_fluor)
      position_fluor = np.array([0,0,0])
      time_fluor     = 0
      ALIVE = False
    elif event == Nsites: #rotate dipole
      local.reorient_dipoles()
    else: #hopping
      #if hopping, update event  
      i = event
  #end of while
  T_int   = np.linspace(np.array(T).min(), np.array(T).max(), 500)
  MSD_int = np.interp(T_int, T, MSD)
  return T_int,MSD_int,MSD_f,D, position_lis,MSD,T 


#Teste 

RF = 1.3
tau = 1
r0=1
n=10
L =  r0*np.array([n])
PBC = [0]
LOAD_FIXED_KAPPA= False
FIXED_KAPPA = 0.476#2/3
save = False
cutoff = r0
nrounds = 1
krot_dip = 0

syst = ld.System(save=save,RF=RF,tau=tau,r0=r0,cutoff=cutoff,L=L,LOAD_FIXED_KAPPA=LOAD_FIXED_KAPPA,FIXED_KAPPA=FIXED_KAPPA,n=n,PBC=PBC,krot_dip=krot_dip)
syst.make_lattice()
results = kmc_run_fluor(syst)

positions = results[4] #arrays das posições do éxciton na rede 
MSD = results[5] #MSD para cada tempo
time = results[6] #tempos 
T_int = results[0]
MSD_int = results[1]

print(T_int)
print('MSD:',MSD_int)
print(positions)
print(MSD)
print(time)
print(len(positions), len(MSD), len(time))

