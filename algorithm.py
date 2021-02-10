# algorithm.py

import sys
import time
import numpy as np
import scipy.ndimage as ndimage

from utils import *

def factorize(data, sigma, n_eigenvectors, n_factors, eig_crit, sim_crit,
                  uniform=True, K=0, seed=255, exclude_eigs=None, verbose=False):
  '''
  an algorithm to factorize a product manifold

  data: an n x d numpy array containing the original data, where n is
    the number of samples and d is the dimension of the data
  sigma: the width of the kernel
  n_eigenvectors: the number of eigenvectors to compute
  n_factors: the number of factors
  eig_crit: the threshold for the eigenvalue criterion
  sim_crit: the threshold for the similarity criterion
  uniform: set to True if the data was sampled uniformly
  K: the voting threshold
  '''
  result = {}
  result['data'] = data
  np.random.seed(seed)

  # part 1: create the data graph and compute eigenvectors, eigenvalues
  if verbose:
    print("\nComputing eigenvectors...")
  t0 = time.perf_counter()
  W = calc_W(data, sigma)
  phi, Sigma = calc_vars(data, W, sigma,
                         n_eigenvectors=n_eigenvectors, uniform=uniform)
  t1 = time.perf_counter()
  if verbose:
    print("  Time: %2.2f seconds" % (t1-t0))

  result['phi'] = phi
  result['Sigma'] = Sigma

  # part 2: searching for reliable triplets (combinations)
  if verbose:
    print("\nComputing combos...")
  t0 = time.perf_counter()
  best_matches, best_sims, all_sims = find_combos(phi, Sigma, n_factors, eig_crit, sim_crit, exclude_eigs=exclude_eigs)
  t1 = time.perf_counter()
  if verbose:
    print("  Time: %2.2f seconds" % (t1-t0))
  result['best_matches'] = best_matches
  result['best_sims'] = best_sims
  result['all_sims'] = all_sims

  # part 3: identifying separate factor manifolds by eigenvectors
  if verbose:
    print("\nSplitting eigenvectors...")
  t0 = time.perf_counter()
  labels, C = split_eigenvectors(best_matches, best_sims, n_eigenvectors, K,
                                 n_factors=n_factors, verbose=verbose)
  t1 = time.perf_counter()
  result['C_matrix'] = C
  if verbose:
    print("  Time: %2.2f seconds" % (t1-t0))

  print("\nManifolds...")
  manifolds = []
  for m in range(n_factors):
    manifold = labels[0][np.where(labels[1]==m)[0]]
    manifolds.append(manifold)
    print("Manifold #{}".format(m + 1), manifold)
  result['manifolds'] = manifolds

  return result
