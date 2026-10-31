#################################################
# KMC for random dipoles and dynamical sampling
# set - 2026
# Authors: Nara Saenger, Tiago Cassiano
################################################
import numpy as np
import scipy
import pandas as pd
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import lattice
from copy import copy
reload_loading = True
if reload_loading:
  import loading as ld


##########################################################################
def process_results(results,dim,RF,tau,r0,FIXED_KAPPA,file_header=""):
  Ts          = np.mean([run[0] for run in results],axis=0)
  MSDs        = np.mean([run[1] for run in results],axis=0)
  D_traj      = np.array([ MSDs[i]/(2*dim*Ts[i]) for i in range(len(MSDs)) ])

  Ts_final_rounds = np.array([run[0][-1] for run in results])
  MSDs_final_rounds = np.array([run[1][-1] for run in results])
  D_final_rounds      = np.array([MSDs_final_rounds[i]/(2*dim*Ts_final_rounds[i]) for i in range(len(MSDs_final_rounds)) ])
  erro_D = np.std(D_final_rounds)*dim

  D_nokappa=(1/tau)*(((RF)**6)/(r0**4))
  kappa2_D_final = D_traj[-1]/D_nokappa
  kappa2_D_final_list_rounds = D_final_rounds/D_nokappa
  
  LD_nokappa      = ((RF)**3) /(r0**2)
  
  
  kappa2          = FIXED_KAPPA
  LD_D_final        = np.sqrt(D_traj[-1]*tau)
  LD_ANALYTIC_1D  = (((kappa2)**(1/2))*(RF)**3) /(r0**2)
  D_ANALYTIC_1D   = (1/tau)*(((kappa2)*(RF)**6)/(r0**4))
  
  
  T_int   = np.linspace(Ts.min(), Ts.max(), 500)
  MSD_int = np.interp(T_int, Ts, MSDs)

  #print(f'MSD de um dos éxcitons = {results[0][1]}'  )

  #print(f"D_init = {D_traj[0]:.2f}")
  print(f"MSD_init = {MSD_int[0]:.2f} MSD_final = {MSD_int[-1]:.2f}")
  print(f"Dana = {D_ANALYTIC_1D:.2f} Dfinal = {D_traj[-1]:.2f}")
  print(f"LDana ={LD_ANALYTIC_1D:.2f} LDcalc={LD_D_final:.2f}")
  print(f"kappa2 ={kappa2_D_final:.2f}")

  out_dic = {}
  out_dic['D']  = [D_ANALYTIC_1D,D_traj[-1]] 
  out_dic['LD'] = [LD_ANALYTIC_1D,LD_D_final]
  out_dic['kappa2_D_final'] = [kappa2_D_final]
  out_dic['MSD_list_mean'] = [MSDs]
  out_dic['T_list_mean'] = [Ts]
  out_dic['kappa2_list_rounds'] = [kappa2_D_final_list_rounds]
  out_dic['erro_D'] = erro_D
  return out_dic





#######################################
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
  local.nsteps_max = syst.nsteps_max


  rates         = local.rate_adj_mat
  probs         = local.prob_adj_mat
  ratesums      = local.ratesum
  distvec       = local.Adj_mat_distvec
  nsteps_max = local.nsteps_max
  #initialization
  Nsites         = local.Nsites
  i              = int(Nsites*np.random.uniform(0,1))
  position       = np.array([0,0,0])
  time           = 0 
  position_fluor = np.array([0,0,0])
  time_fluor     = 0
  step = 0

  
  MSD         = []
  T           = []
  MSD_f          = []
  D           = []
  fluor_pos   = []
  fluor_times = []
  #steps = []
  ALIVE       = True


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
    elif event == Nsites: #rotate dipole
      local.reorient_dipoles()
    else: #hopping
      #if hopping, update event  
      i = event

    step +=1 
    if step > nsteps_max: 
      ALIVE = False
  #end of while

  T_int   = np.linspace(np.array(T).min(), np.array(T).max(), 500)
  MSD_int = np.interp(T_int, T, MSD)
  return T_int,MSD_int,MSD_f,D



def kmc_run_nofluor(syst):
  local               = copy(syst)
  local.dipoles       = syst.dipoles.copy()
  local.kappa_adj_mat = syst.kappa_adj_mat.copy()
  local.rate_adj_mat  = syst.rate_adj_mat.copy()
  local.prob_adj_mat  = syst.prob_adj_mat.copy()
  local.ratesum       = syst.ratesum.copy()
  local.prob_dirty    = syst.prob_dirty.copy()
  local.nsteps_max = syst.nsteps_max

  rates         = local.rate_adj_mat
  probs         = local.prob_adj_mat
  ratesums      = local.ratesum
  distvec       = local.Adj_mat_distvec
  #initialization
  Nsites         = local.Nsites
  nsteps_max = local.nsteps_max
  i              = int(Nsites*np.random.uniform(0,1))
  step = 0
  position       = np.array([0,0,0])
  time           = 0 


  MSD         = []
  T           = []
  D           = []
  
  for step in range(nsteps_max): #dá um passo até um determinado número de vezes 
    step += 1
    if local.prob_dirty[i]: #if this particular site was involved in a previous dipole reorientation
        ld.update_prob_row(i,rates,probs,ratesums)
        local.prob_dirty[i] = False #now the prob is correct. Prob is nt dirty anymore
    probs_i = probs[i,:]
    ratesum = ratesums[i]
    
    u = np.random.uniform(0,1)
    event = np.searchsorted(probs_i, u, side="right")
    
    shift    = distvec[i,event]
    position = position + shift 
    
    dT         = -np.log(np.random.uniform(0,1))/ratesum
    time       = time  + dT
    
    MSD.append(np.dot(position,position))
    T.append(time)    

    if event == Nsites: #rotate dipole
      local.reorient_dipoles()
    else: #hopping
      #if hopping, update event  
      i = event
    
  T_int   = np.linspace(np.array(T).min(), np.array(T).max(), 500)
  MSD_int = np.interp(T_int, T, MSD)
  return T_int,MSD_int
  


# good metrics
#nrounds = 1000
#nsteps  = 100000

#nrounds = 100

#results = Parallel(n_jobs=10,verbose=10)(delayed(kmc_run_fluor)(rates, probs, distvec, dist,dim) for n in range(nrounds))
#process_results(results)  
  
  
  
  
  
  
  
  
  
  
  
  

