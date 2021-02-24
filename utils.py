# helper methods for the algorithm.

import numpy as np
import sys
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cvxpy as cp

from itertools import combinations
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.extmath import randomized_svd
from mpl_toolkits.mplot3d import Axes3D

def get_gt_data(data, datatype):
  '''
  converts the observed data to the ground truth data from the latent manifold
  '''
  if datatype == 'line_circle':
    data_gt = np.zeros((data.shape[0],2))
    data_gt[:,0] = data[:,0]
    data_gt[:,1] = np.arctan2(data[:,2], data[:,1])
  elif datatype == 'rectangle3d':
    data_gt = data[:,:2]
  elif datatype == 'torus':
    data_gt = np.zeros((data.shape[0],2))
    data_gt[:,0] = np.pi + np.arctan2(data[:,1] - np.mean(data[:,1]),
                                      data[:,0] - np.mean(data[:,0]))
    data_gt[:,1] = np.pi + np.arctan2(data[:,3] - np.mean(data[:,3]),
                                      data[:,2] - np.mean(data[:,2]))
  else:
    data_gt = data

  return data_gt

def downsample_data(data, l):
  '''
  downsamples data with resolution L * L to resolution l * l
  '''
  stride = L // l
  lowres_data = data[:,::stride,::stride]
  return lowres_data

###
# COMPUTE EIGENVECTORS
###

def calc_W(data, sigma):
  '''
  calculates the weight matrix W
  data: the observed data stored as an n x D matrix, where n is the number of
        samples and D is the dimension of the data space
  sigma: the kernel width
  '''

  pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))
  W = np.exp(-pairwise_sq_dists / sigma)
  return W

def calc_vars(data, W, sigma, n_eigenvectors):
  '''
  calculates phi and Sigma
  data: the observed data stored as an n x D matrix, where n is the number of
         samples and D is the dimension of the data space
  W: the weight matrix of the data
  sigma: the kernel width
  n_eigenvectors: the number of eigenvectors to compute
  '''
  ones = np.ones(W.shape[0])
  v = np.sqrt(W @ ones)
  S = W / np.outer(v, v)
  V, Sigma, VT = randomized_svd(S,
                                n_components=n_eigenvectors,
                                n_iter=5,
                                random_state=None)
  phi = V / V[:,0][:,None]
  Sigma = -np.log(Sigma) / sigma
  return phi, Sigma

###
# FIND BEST EIGENVECTOR COMBOS
###

def calculate_sim(v_i, v_j):
  '''
  calculates the similarity score between vectors v_i and v_j
  S(v_i, v_j) = < v_i/||v_i||, v_j/||v_j|| >
  where S is the "similarity" function described in the paper
  '''

  # normalize vectors to unit norm
  v_i /= np.linalg.norm(v_i)
  v_j /= np.linalg.norm(v_j)

  # calculate L2 distance between v1 and v2
  sim = np.dot(v_i, v_j)
  return sim # score

def find_combos(phi, Sigma, n_factors=2, eig_crit=10e-3, sim_crit=0.5, exclude_eigs=None):
  '''
  computes the triplets which have the highest similarity scores
  returns:
  best_matches: a dictionary of triplets indexed by the product eigenvector
  max_sims: a dictionary of the triplet correlations indexed by the product eigenvector
  all_sims: all of the correlations for each product eigenvector 1...n_eigenvectors
  '''
  best_matches = {}
  max_sims = {}
  all_sims = {}

  for k in range(2, phi.shape[1]): # k is the proposed product eigenvector
    # show progress
    if (k % 10 == 0):
      sys.stdout.write('\r')
      sys.stdout.write("[%-20s] %d%%" % ('='*int(20*k/phi.shape[1]), 100*k/phi.shape[1]))
      sys.stdout.flush()

    v_k = phi[:,k]
    lambda_k = Sigma[k]
    max_sim = 0
    best_match = []

    # iterate over all possible number of factors in the eigenvector factorization
    for m in range(2, n_factors + 1):
      # iterate over all possible factorizations
      if exclude_eigs is not None:
        valid_eigs = [v for v in np.arange(1, k) if v not in exclude_eigs]
      else:
        valid_eigs = np.arange(1, k)
      for combo in list(combinations(valid_eigs, m)):
        combo = list(combo)
        lambda_sum = np.sum(Sigma[combo])
        lambda_diff = abs(lambda_k - lambda_sum)
        if lambda_diff < eig_crit:
          # get product of proposed base eigenvectors
          v_combo = np.ones(phi.shape[0])
          for i in combo:
            v_combo *= phi[:,i]

          # test with positive
          sim = calculate_sim(v_combo, v_k)
          if sim > max_sim:
            best_match = combo
            max_sim = sim

          # test with negative
          dsim = calculate_sim(v_combo, -v_k)
          if sim > max_sim:
            best_match = combo
            max_sim = sim

    if len(best_match) > 0:
      all_sims[k] = max_sim
      if max_sim >= sim_crit:
        best_matches[k] = list(best_match)
        max_sims[k] = max_sim

  return best_matches, max_sims, all_sims

###
# VOTING SCHEME
###

def split_eigenvectors(best_matches, best_sims, n_eigenvectors, K=0,
                       n_factors=2, verbose=False):
  '''
  clusters eigenvectors into two separate groups
  returns:
  labels: a 2 x m array where m is the number of factor eigenvectors identified
          the first row of the array contains the indices of each factor eigenvector
          the second row of the array contains the label of the manifold factor
          to which the factor eigenvector corresponds
  C: the separability matrix
  '''
  votes = np.zeros(n_eigenvectors)
  C = np.zeros((n_eigenvectors, n_eigenvectors))

  if verbose:
    print('{:10} {:>7} {:>12}'.format('Combo', 'Match', 'Similarity'))

  for match in list(best_matches):
    combo = best_matches[match]
    if verbose:
      print('{:10} {:7} {:12}'.format(str(combo), match, np.around(best_sims[match], 3)))
    for pair in list(combinations(combo, 2)):
      C[pair[0]][pair[1]] += best_sims[match]
      C[pair[1]][pair[0]] += best_sims[match]
      votes[pair[0]] += best_sims[match]
      votes[pair[1]] += best_sims[match]

  # perform spectral clustering based on independent vectors
  factors = np.where(votes>0)[0]
  n = len(factors)
  C_ = np.ones((n, n))
  for i in range(C_.shape[0]):
    for j in range(C_.shape[1]):
      if i == j:
        C_[i][j] = 0
      else:
        C_[i][j] = C[factors[i]][factors[j]]

  if verbose:
    print("\nSeparability matrix:\n", np.around(C_, 3))

  np.random.seed(1)
  if n_factors == 2:
    Y = cp.Variable((n, n), PSD=True)
    constraints = [cp.diag(Y) == 1]
    obj = 0.5 * cp.sum(cp.multiply(C_, np.ones((n, n)) - Y))
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve()

    eigenvalues, eigenvectors = np.linalg.eigh(Y.value)
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T
    partition = np.random.normal(size=n)
    projections = assignment.T @ partition

    labels = np.zeros((2, n), dtype='int')
    labels[0,:] = factors
    for i in range(n):
      labels[1][i] = 1 if projections[i] >= 0 else 0

  elif n_factors > 2:
    # follow max k-cut heuristic according to:
    # https://drops.dagstuhl.de/opus/volltexte/2018/8309/pdf/OASIcs-SOSA-2018-13.pdf

    Y = cp.Variable((n,n), PSD=True)
    constraints = [cp.diag(Y) == 1]
    for i in range(n):
      for j in range(n):
        if j is not i:
          constraints += [Y[i,j] >= -1 / (n_factors - 1)]

    obj = (1 - 1 / n_factors) * cp.sum(cp.multiply(C_, np.ones((n, n)) - Y))
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve()

    eigenvalues, eigenvectors = np.linalg.eigh(Y.value)
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T # v_i
    assignment_ = np.concatenate((np.zeros(assignment.shape), assignment), axis=0) # (0, v_i)
    assignment = np.concatenate((assignment, np.zeros(assignment.shape)), axis=0) # (v_i, 0)
    g = np.random.normal(size=(2 * n))
    g /= np.linalg.norm(g)
    g_assignment = assignment.T @ g
    g_assignment_ = assignment_.T @ g
    theta = np.arctan2((g_assignment_), (g_assignment))
    z = 2 * np.pi * np.random.random()

    labels = np.zeros((2, n), dtype='int')
    labels[0,:] = factors
    for i in range(n):
      labels[1][i] = int(((theta[i] - z) % (2 * np.pi)) / (2 * np.pi / n_factors))

  return labels, C

def get_product_eigs(manifolds, n_eigenvectors):
  '''
  returns the a list of product eigenvectors out of the first n_eigenvectors,
  given a manifold factorization
  manifolds: a list of lists, each sublist contains the indices of factor
             eigenvectors corresponding to a manifold factor
  n_eigenvectors: the number of eigenvectors
  '''
  mixtures = []
  for i in range(1, n_eigenvectors):
    is_mixture = True
    for manifold in manifolds:
      if i in manifold:
        is_mixture = False
    if is_mixture:
      mixtures.append(i)

  return mixtures
