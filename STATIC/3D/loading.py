#################################################
# KMC for random dipoles and dynamical sampling
# set - 2026
# Authors: Nara Saenger, Tiago Cassiano
################################################
import numpy as np
import scipy
import pandas as pd
import lattice
#####################################

# calculate distances
#adjancency matrix: a[i,j] = dist(r_i,r_j) = | r_j-r_i |


def dist_sites(L,PBC,coord_i,coord_j):
  diff =  np.array(coord_j)-np.array(coord_i)
  for i in range(3):
    if i in PBC:
      delt = diff[i]
      diff[i] = delt - L[i]*np.round(delt/L[i])
  diff_dot = np.dot(diff,diff)
  distance = np.sqrt(diff_dot)
  if distance > 0:
    norm_vec = diff/distance
  else:
    norm_vec = np.array([0,0,0]) #diagonal terms in adj mat
  return distance, norm_vec

def make_adj_mat(L,PBC,Nsites,coord,save=True):  
  Adj_mat          = np.zeros([Nsites,Nsites+1])
  Adj_mat_normvec  = np.zeros([Nsites,Nsites+1,3])
  Adj_mat_distvec  = np.zeros([Nsites,Nsites+1,3])
  for i in range(Nsites):
    for j in range(Nsites):
      dist, normvec         = dist_sites(L,PBC,coord[i],coord[j])
      Adj_mat[i,j]          = dist
      Adj_mat_normvec[i,j]  = normvec
      #print(dist,normvec)
      Adj_mat_distvec[i,j]  = dist*normvec
  if save:
    np.savetxt(adjmat,Adj_mat)
    np.save(adjmat_normvec, Adj_mat_normvec)
    np.save(adjmat_distvec, Adj_mat_distvec)
  return Adj_mat,Adj_mat_normvec,Adj_mat_distvec
  

#######################################
## calculate kappas
#load dipole
def calc_kappa(e,mu_i,mu_j):
  norm1 = np.sqrt(np.dot(mu_i,mu_i))
  norm2 = np.sqrt(np.dot(mu_j,mu_j))  
  kappa = np.dot(mu_i,mu_j) - 3*(np.dot(mu_i,e))*(np.dot(mu_j,e))
  kappa = kappa/(norm1*norm2)
  return kappa**2
def update_row_kappa(LOAD_FIXED_KAPPA,FIXED_KAPPA,Nsites,indx_i,new_dip,dipoles,Adj_mat_normvec):
  kappa_adj_mat_row =np.zeros([1,Nsites])
  mu_i = new_dip
  for j in range(Nsites):
    e_ij = Adj_mat_normvec[indx_i,j]
    mu_j = dipoles[j]
    if LOAD_FIXED_KAPPA:
      kappa_adj_mat_row[0,j] = FIXED_KAPPA
    else:
      kappa_adj_mat_row[0,j] = calc_kappa(e_ij,mu_i,mu_j)
  return  kappa_adj_mat_row   
def calc_kappas(LOAD_FIXED_KAPPA,FIXED_KAPPA,Nsites,dipoles,Adj_mat_normvec,save=True):
  #calculate kappa matrix following the same notation of Adj_mat
  kappa_adj_mat =np.zeros([Nsites,Nsites])
  for i in range(Nsites):
    for j in range(Nsites):
      e_ij = Adj_mat_normvec[i,j]
      mu_i = dipoles[i]
      mu_j = dipoles[j]
      if LOAD_FIXED_KAPPA:
        kappa_adj_mat[i,j] = FIXED_KAPPA
      else:
        kappa_adj_mat[i,j] = calc_kappa(e_ij,mu_i,mu_j)
  if save:
    np.savetxt(adjmat_kappa,kappa_adj_mat)
  print(f'this program run with <Kappa^2> = {np.mean(kappa_adj_mat.flatten()):.5f}')
  return kappa_adj_mat
#######################################
#Calculate rates
def fret_rate(kappa2,dist,Rf,tau,r0):
  return (1/tau)*(kappa2)*((Rf/(dist))**6)
'''  
def calc_rates_and_probs(tau,RF,r0,cutoff,Nsites,Adj_mat,kappa_adj_mat,save=True):
  rate_adj_mat =np.zeros([Nsites,Nsites])
  prob_adj_mat =np.zeros([Nsites,Nsites])
  for i in range(Nsites):
    for j in range(Nsites):
      dist  = Adj_mat[i,j]
      if ((i!=j) and (dist<=cutoff)):
        kappa2 = kappa_adj_mat[i,j]
        rate_adj_mat[i,j]  = fret_rate(kappa2,dist,RF,tau,r0)   
  ratesum = np.zeros([Nsites])    
  for i in range(Nsites):
    rate_adj_mat[i,i] = 1/tau #entry (i,i) is fluorescence
    prob_adj_mat[i,:] = rate_adj_mat[i,:]/np.sum(rate_adj_mat[i,:]) #converting into prob
    prob_adj_mat[i,:] = np.cumsum(prob_adj_mat[i,:]) #converting into cumsum
    prob_adj_mat[i,:] = prob_adj_mat[i,:]/prob_adj_mat[i,-1]
    ratesum[i] = np.sum(rate_adj_mat[i,:])
  if save:
    np.savetxt(rate_mat,rate_adj_mat)
    np.savetxt(prob_mat,prob_adj_mat)
    np.savetxt(ratesum_vec,ratesum)
  return rate_adj_mat,prob_adj_mat,ratesum
'''  
def calc_rates_and_probs(rate_dipole,tau,RF,r0,cutoff,Nsites,Adj_mat,kappa_adj_mat,save=True):
  rate_adj_mat =np.zeros([Nsites,Nsites+1])
  prob_adj_mat =np.zeros([Nsites,Nsites+1])
  for i in range(Nsites):
    for j in range(Nsites):
      dist  = Adj_mat[i,j]
      if ((i!=j) and (dist<=cutoff)):
        kappa2 = kappa_adj_mat[i,j]
        rate_adj_mat[i,j]  = fret_rate(kappa2,dist,RF,tau,r0)
        
      rate_adj_mat[i,Nsites] = rate_dipole  #entry (i,Nsites) is dipole re-orientation
      rate_adj_mat[i,i] = 1/tau #entry (i,i) is fluorescence
  ratesum = np.zeros([Nsites+1])    
  for i in range(Nsites):
    ratesum[i] = np.sum(rate_adj_mat[i,:])
    prob_adj_mat[i,:] = rate_adj_mat[i,:]/ratesum[i] #converting into prob
    prob_adj_mat[i,:] = np.cumsum(prob_adj_mat[i,:]) #converting into cumsum
    prob_adj_mat[i,:] = prob_adj_mat[i,:]/prob_adj_mat[i,-1]
  if save:
    np.savetxt(rate_mat,rate_adj_mat)
    np.savetxt(prob_mat,prob_adj_mat)
    np.savetxt(ratesum_vec,ratesum)
  return rate_adj_mat,prob_adj_mat,ratesum

def update_rates_after_rotation(chosen, tau, RF, r0, cutoff,Nsites, Adj_mat, kappa_adj_mat,rate_adj_mat, ratesum):
    affected = [chosen]
    for j in range(Nsites):
        if j == chosen: #diagonal terms are not affected because fluorescence is orientational-independent in this model
            continue
        dist = Adj_mat[chosen,j]

        if dist <= cutoff:
            old_rate = rate_adj_mat[chosen,j]
            kappa2 = kappa_adj_mat[chosen,j] #new kappa, updated earlier
            new_rate = fret_rate(kappa2, dist, RF, tau, r0)
            # chosen -> j
            rate_adj_mat[chosen,j] = new_rate
            # j -> chosen
            rate_adj_mat[j,chosen] = new_rate
            # only one entry changed in row j
            ratesum[j] += new_rate - old_rate
            affected.append(j)

    # Several entries changed in chosen row.
    ratesum[chosen] = np.sum(rate_adj_mat[chosen,:])
    return affected
def update_prob_row(i, rate_adj_mat, prob_adj_mat, ratesum):
    prob_adj_mat[i,:] = np.cumsum(rate_adj_mat[i,:]) / ratesum[i]


class System:
    def __init__(self,save,krot_dip,RF,tau,r0,cutoff,L,LOAD_FIXED_KAPPA,FIXED_KAPPA,PBC,n):
        self.save   = save
        self.RF     = RF
        self.tau    = tau
        self.r0     = r0
        self.cutoff = cutoff
        self.L      = L
        self.LOAD_FIXED_KAPPA = LOAD_FIXED_KAPPA
        self.FIXED_KAPPA = FIXED_KAPPA
        self.krot_dip    = krot_dip
        self.n    = n
        self.PBC  = PBC
        self.readlatt       = 'lattice.txt'
        self.adjmat         = 'adjmat.txt'
        self.adjmat_normvec = 'adjmatvec.npy'
        self.readdipole     = 'dipoles.txt'
        self.adjmat_kappa   = 'kappa_mat.txt'
        self.rate_mat       = 'rates.txt'
        self.adjmat_distvec = 'adjmat_distances_vec.npy'
        self.prob_mat       = 'prob.txt'
        self.ratesum_vec    = 'ratesum.txt'
    def make_lattice(self,loadcoord=False,loaddip=False,loadrates=False):
        if loadcoord:
            coord     = np.loadtxt(readlatt)
        else:
            coord     = lattice.make_regular_lattice(r0=self.r0,L=self.L,n=self.n,save=self.save)
        self.Nsites = len(coord)
        if loaddip:
            dipoles   = np.loadtxt(readdipole)
        else:
            dipoles = lattice.make_random_dipoles(self.Nsites,save=self.save)
        self.dipoles = dipoles
        if loadrates:
            self.rate_adj_mat     = np.loadtxt(rate_mat)
            self.prob_adj_mat     = np.loadtxt(prob_mat)
            self.Adj_mat_distvec  = np.load(adjmat_distvec)
            self.ratesum          = np.loadtxt(ratesum_vec)
            self.Adj_mat          = np.loadtxt(adjmat)
        else:    
            #making adjacency matrix
            Adj_mat,Adj_mat_normvec,Adj_mat_distvec = make_adj_mat(self.L,self.PBC,self.Nsites,coord,save=self.save)
            #calculating kappas
            kappa_adj_mat = calc_kappas(self.LOAD_FIXED_KAPPA,self.FIXED_KAPPA,self.Nsites,dipoles,Adj_mat_normvec,save=self.save)
            #calculate rates and probs
            rate_adj_mat,prob_adj_mat,ratesum = calc_rates_and_probs(self.krot_dip,self.tau,self.RF,self.r0,self.cutoff,self.Nsites,Adj_mat,kappa_adj_mat,save=self.save)        
        #adding to the memory
        self.prob_dirty = np.zeros(self.Nsites,dtype=bool)
        self.Adj_mat         = Adj_mat
        self.Adj_mat_normvec = Adj_mat_normvec
        self.Adj_mat_distvec = Adj_mat_distvec
        self.kappa_adj_mat   = kappa_adj_mat
        self.rate_adj_mat    = rate_adj_mat
        self.prob_adj_mat    = prob_adj_mat
        self.ratesum         = ratesum
        print('DONE LOADING LATTICE')
    def reorient_dipoles(self):
        chosen        = int(self.Nsites*np.random.uniform(0,1))
        new_dipole    = lattice.make_random_dipole()
        self.dipoles[chosen] =new_dipole
        new_row_kappa = update_row_kappa(self.LOAD_FIXED_KAPPA,self.FIXED_KAPPA,self.Nsites,chosen,new_dipole,self.dipoles,self.Adj_mat_normvec)         
        self.kappa_adj_mat[chosen,:] = new_row_kappa
        self.kappa_adj_mat[:,chosen] = new_row_kappa
        # Update only raw rates actually affected
        affected = update_rates_after_rotation(chosen,self.tau,self.RF,self.r0,self.cutoff,self.Nsites,self.Adj_mat,self.kappa_adj_mat,self.rate_adj_mat,self.ratesum)#updating the rates, probs
        # Their stored probability rows are now stale
        self.prob_dirty[affected] = True      





