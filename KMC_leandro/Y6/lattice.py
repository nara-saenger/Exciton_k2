#################################################
# KMC for random dipoles and dynamical sampling
# set - 2026
# Authors: Nara Saenger, Tiago Cassiano
################################################

import numpy as np
import scipy
import pandas as pd


def make_regular_lattice(r0,L,n,save=True):
  dim = len(L)
  X  = r0*np.arange(n)
  Y  = r0*np.arange(n) #[0]
  Z  = r0*np.arange(n)  #[0]

  latt = []
  if dim == 3:
    for x in X:
      for y in Y:
        for z in Z:
          latt.append([x,y,z])
  if dim == 2:
    for x in X:
      for y in Y:
        latt.append([x,y,0])
  if dim == 1:
    for x in X:
      latt.append([x,0,0])                
  if save:
    np.savetxt(readlatt,latt)
  return latt
#random dipoles
def make_random_dipole():
    x = np.random.uniform(-1,1)
    y = np.random.uniform(-1,1)
    z = np.random.uniform(-1,1)
    norm = (x**2 + y**2 + z**2)**0.5
    x = x/norm
    y = y/norm
    z = z/norm
    return [round(x,5),round(y,5),round(z,5)]  
def make_random_dipoles(Nsites,save=True):
  dipoles = [make_random_dipole() for i in range(Nsites)]
  if save:
    np.savetxt(readdipole,np.array(dipoles))
  return dipoles

