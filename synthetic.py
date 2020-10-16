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

from plots import *

###
# synthetic.py
#
# Generate synthetic random data sampled from a toy product manifold and compute
# the Laplacian, eigenvalues, and independent manifolds of the data
###

def generate_synthetic_data(dimensions, n_samples, datatype='rectangle',
                            seed=0, noise=0.05):
  '''
  generates uniform random data from simple geometric manifolds
  dimensions: the dimensions of the data
  num_samples: the number of samples to generate
  seed: the random seed for generating the data
  type: the type of data to generate
  '''

  np.random.seed(seed)

  if datatype=='rectangle':
    # rectangle
    l1, l2 = dimensions
    line1_data = l1 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line2_data = l2 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    data = np.column_stack((line1_data, line2_data))

  elif datatype=='rectangle3d':
    # rectangle
    l1, l2, z_noise = dimensions
    line1_data = l1 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line2_data = l2 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line3_data = np.random.normal(scale=z_noise, size=n_samples)
    data = np.column_stack((line1_data, line2_data, line3_data))

  elif datatype=='line_circle':
    # hollow cylinder
    l1, l2 = dimensions
    line_data = l1 * (np.random.rand(n_samples) +  \
                      np.random.normal(scale=noise, size=n_samples))
    circle_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle_data[:,0] = (l2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle_data[:,1] = (l2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    data = np.column_stack((line_data, circle_data))

  elif datatype=='cube':
    # cube
    l1, l2, l3 = dimensions
    line1_data = l1 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line2_data = l2 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line3_data = l3 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    data = np.column_stack((line1_data, line2_data, line3_data))

  elif datatype=='torus':
    #torus
    r1, r2 = dimensions
    circle1_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle1_data[:,0] = r1 + (r1 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle1_data[:,1] = r1 + (r1 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    circle2_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle2_data[:,0] = r2 + (r2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle2_data[:,1] = r2 + (r2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    data = np.column_stack((circle1_data, circle2_data))

  else:
    print('Error: invalid data type')
    return

  return data

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
    data_gt[:,0] = np.arctan2(data[:,1] - np.mean(data[:,1]),
                              data[:,0] - np.mean(data[:,0]))
    data_gt[:,1] = np.arctan2(data[:,3] - np.mean(data[:,3]),
                              data[:,2] - np.mean(data[:,2]))
  else:
    data_gt = data

  return data_gt

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

def calc_vars(data, W, sigma, n_eigenvectors, uniform=True):
  '''
  calculates phi and Sigma
  data: the observed data stored as an n x D matrix, where n is the number of
         samples and D is the dimension of the data space
  W: the weight matrix of the data
  sigma: the kernel width
  n_eigenvectors: the number of eigenvectors to compute
  uniform: set to False if the data was not sampled uniformly
  '''
  ones = np.ones(W.shape[0])
  if uniform:
    v = np.sqrt(W @ ones)
    S = W / np.outer(v, v)
    V, Sigma, VT = randomized_svd(S,
                                  n_components=n_eigenvectors,
                                  n_iter=5,
                                  random_state=None)
    phi = V / V[:,0][:,None]
    Sigma = -np.log(Sigma) / sigma
  else:
    p = W @ ones
    W1 = W / np.outer(p, p)
    v = np.sqrt(W1 @ ones)
    W2 = W1 / np.outer(v, v)
    V, Sigma, VT = randomized_svd(W2,
                                  n_components=n_eigenvectors,
                                  n_iter=5,
                                  random_state=None)
    phi = V / V[:,0][:,None]
  return phi, Sigma

###
# FIND BEST EIGENVECTOR COMBOS
###

def calculate_corr(v_i, v_j):
  '''
  calculates the correlation between vectors v_i and v_j

  S(v_i, v_j) = < v_i/||v_i||, v_j/||v_j|| >
  where S is the "similarity" function described in the paper
  '''

  # normalize vectors to unit norm
  v_i /= np.linalg.norm(v_i)
  v_j /= np.linalg.norm(v_j)

  # calculate L2 distance between v1 and v2
  corr = np.dot(v_i, v_j)
  return corr # score

def find_combos(phi, Sigma, n_comps=2, lambda_thresh=10e-3, corr_thresh=0.5):
  '''
  computes the triplets which have the highest similarity scores

  returns:
  best_matches: a dictionary of triplets indexed by the product eigenvector
  max_corrs: a dictionary of the triplet correlations indexed by the product eigenvector
  all_corrs: all of the correlations for each product eigenvector 1...n_eigenvectors
  '''
  best_matches = {}
  max_corrs = {}
  all_corrs = {}

  for k in range(2, phi.shape[1]): # k is the proposed product eigenvector
    # show progress
    if (k % 10 == 0):
      sys.stdout.write('\r')
      sys.stdout.write("[%-20s] %d%%" % ('='*int(20*k/phi.shape[1]), 100*k/phi.shape[1]))
      sys.stdout.flush()

    v_k = phi[:,k]
    lambda_k = Sigma[k]
    max_corr = 0
    best_match = []

    # iterate over all possible number of factors in the eigenvector factorization
    for m in range(2, n_comps + 1):
      # iterate over all possible factorizations
      for combo in list(combinations(np.arange(1, k), m)):
        combo = list(combo)
        lambda_sum = np.sum(Sigma[combo])
        lambda_diff = abs(lambda_k - lambda_sum)
        if lambda_diff < lambda_thresh:
          # get product of proposed base eigenvectors
          v_combo = np.ones(phi.shape[0])
          for i in combo:
            v_combo *= phi[:,i]

          # test with positive
          corr = calculate_corr(v_combo, v_k)
          if corr > max_corr:
            best_match = combo
            max_corr = corr

          # test with negative
          dcorr = calculate_corr(v_combo, -v_k)
          if corr > max_corr:
            best_match = combo
            max_corr = corr

    if len(best_match) > 0:
      all_corrs[k] = max_corr
      if max_corr >= corr_thresh:
        best_matches[k] = list(best_match)
        max_corrs[k] = max_corr

  return best_matches, max_corrs, all_corrs

###
# VOTING SCHEME
###

def split_eigenvectors(best_matches, best_corrs, n_eigenvectors, K,
                       n_comps=2, verbose=False):
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
    print('{:10} {:>7} {:>7}'.format('Combo', 'Match', 'Corr'))

  for match in list(best_matches):
    combo = best_matches[match]
    if verbose:
      print('{:10} {:7} {:7}'.format(str(combo), match, np.around(best_corrs[match], 3)))
    for pair in list(combinations(combo, 2)):
      C[pair[0]][pair[1]] += best_corrs[match]
      C[pair[1]][pair[0]] += best_corrs[match]
      votes[pair[0]] += best_corrs[match]
      votes[pair[1]] += best_corrs[match]

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
  if n_comps == 2:
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

  elif n_comps > 2:
    # follow max k-cut heuristic according to:
    # https://drops.dagstuhl.de/opus/volltexte/2018/8309/pdf/OASIcs-SOSA-2018-13.pdf

    Y = cp.Variable((n,n), PSD=True)
    constraints = [cp.diag(Y) == 1]
    for i in range(n):
      for j in range(n):
        if j is not i:
          constraints += [Y[i,j] >= -1 / (n_comps - 1)]

    obj = (1 - 1 / n_comps) * cp.sum(cp.multiply(C_, np.ones((n, n)) - Y))
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
      labels[1][i] = int(((theta[i] - z) % (2 * np.pi)) / (2 * np.pi / n_comps))

    # plot_k_cut(labels, n_comps, theta, z)

  return labels, C

def get_mixture_eigenvectors(manifolds, n_eigenvectors):
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
