import time
import sys
import json

import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import SpectralClustering


###
# GENERATE RANDOM DATA
###

def generate_data(l1, l2, n_samples=10000, seed=0, datatype='line_line', noise=0.05):
  '''
  generates uniform random data
  l1: the first length parameter
  l2: the second length parameter
  num_samples: the number of samples to generate
  seed: the random seed for generating the data
  type: the type of data to generate
  '''

  np.random.seed(seed)

  if datatype=='line_line':
    # two line segments
    line1_data = l1 * (np.random.rand(n_samples) + noise * (2 * np.random.rand(n_samples) - 1))
    line2_data = l2 * (np.random.rand(n_samples) + noise * (2 * np.random.rand(n_samples) - 1))
    data = np.column_stack((line1_data, line2_data))

  elif datatype=='line_circle':
    # line segment and circle
    line_data = l1 * (np.random.rand(n_samples) + noise * (2 * np.random.rand(n_samples) - 1))
    circle_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle_data[:,0] = (l2 + noise * (2 * np.random.rand(n_samples) - 1)) * np.cos(theta)
    circle_data[:,1] = (l2 + noise * (2 * np.random.rand(n_samples) - 1)) * np.sin(theta)
    data = np.column_stack((line_data, circle_data))

  elif datatype=='circle_circle':
    # circle and circle
    circleA_data = np.empty((n_samples,2))
    circleB_data = np.empty((n_samples,2))
    for i in range(n_samples):
      theta = 2 * np.pi * np.random.rand()
      circleA_data[i,:] = [(l1 + noise * (2 * np.random.rand(n_samples) - 1)) * np.cos(theta), l1 * np.sin(theta)]
      theta = 2 * np.pi * np.random.rand()
      circleB_data[i,:] = [(l2 + noise * (2 * np.random.rand(n_samples) - 1)) * np.cos(theta), l2 * np.sin(theta)]
    data = np.column_stack((circleA_data, circleB_data))

  elif datatype=='rect_circle':
    # rectangle and circle
    line1_data = l1 * (np.random.rand(n_samples) + noise * (2 * np.random.rand(n_samples) - 1))
    line2_data = l2 * (np.random.rand(n_samples) + noise * (2 * np.random.rand(n_samples) - 1))
    rect_data = np.column_stack((line1_data, line2_data))

    circle_data = np.empty((n_samples,2))
    for i in range(n_samples):
      theta = 2 * np.pi * np.random.rand()
      circle_data[i,:] = [(l1 + noise * (2 * np.random.rand() - 1)) * np.cos(theta), l1 * np.sin(theta)]
    data = np.column_stack((rect_data, circle_data))

  else:
    print('Error: invalid data type')
    return

  return data

###
# COMPUTE EIGENVECTORS
###

def get_sigma(n_samples):
  '''
  calculates an appropriate sigma
  '''
  sigma = 1 / (n_samples**(1 / (n_samples / 2 + 3)))
  return sigma

def calc_W(data, sigma=None):
  '''
  calculates the weight matrix W
  '''

  t0 = time.perf_counter()
  pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))

  if sigma is None:
    sigma = get_sigma(data.shape[0])

  W = np.exp(-pairwise_sq_dists / sigma)
  t1 = time.perf_counter()
  print("  Calculating W took %2.2f seconds" % (t1-t0))

  return W

def calc_vars(data, W, n_comps=100):
  '''
  calculates phi, psi, and Sigma
  '''

  t0 = time.perf_counter()
  ones = np.ones(W.shape[0])
  v = np.sqrt(W @ ones)
  S = W / np.outer(v, v)

  V, Sigma, VT = randomized_svd(S,
                                n_components=n_comps,
                                n_iter=5,
                                random_state=None)
  phi = V / V[:,0][:,None]
  Sigma = -np.log(Sigma) / get_sigma(data.shape[0])

  t1 = time.perf_counter()
  print("  Calculating phi, Sigma took %2.2f seconds" % (t1-t0))

  return phi, Sigma

###
# FIND BEST EIGENVECTOR TRIPLETS
###

def calculate_score(v_i, v_j):
  '''
  calculates proximity of v_i to v_j
  '''

  # normalize vectors to unit norm
  v_i /= np.linalg.norm(v_i)
  v_j /= np.linalg.norm(v_j)

  # calculate L2 distance between v1 and v2
  score = np.linalg.norm((v_i - v_j), ord=2)
  return score

def find_triplets(phi, Sigma, n_comps, lambda_thresh=10e-3):
  best_matches = {}
  min_dists = {}
  for k in range(2, n_comps):
    v_k = phi[:,k]
    lambda_k = Sigma[k]
    min_dist = np.inf
    best_pair = [0, 1]
    for i in range(0, k):
      for j in range(i, k):
        v_i = phi[:,i]
        v_j = phi[:,j]
        v_ij = v_i * v_j
        lambda_ij = Sigma[i] + Sigma[j]
        if abs(lambda_k - lambda_ij) < lambda_thresh:
          # test with positive
          d = calculate_score(v_ij, v_k)
          if d < min_dist:
            best_pair = [i, j]
            min_dist = d

          # test with negative
          d = calculate_score(v_ij, -v_k)
          if d < min_dist:
            best_pair = [i, j]
            min_dist = d

    if 0 not in best_pair:
      best_matches[k] = best_pair
      min_dists[k] = min_dist

  print(best_matches)
  return best_matches, min_dists

###
# VOTING SCHEME
###
def vote(edge_scores, votes, Sigma, triplet, dist):
  '''
  the scoring for scheme 3 is as follows:
  when encountering a new triplet (t1, t2, t3):
  place a negative vote for edge t1 - t2
  (they are likely to be in opposite sets)
  place a positive vote for edges t1 - t1 and t2 - t2
  (they are likely to be eigenvectors of an independent manifold)
  place a negative vote for edge t3 - t3
  (t3 is unlikely to be an eigenvector of an independent manifold)
  '''

  t1, t2, t3 = [int(i) for i in triplet]
  edge_scores[t1][t2] += 1
  edge_scores[t2][t1] += 1

  votes[t1] += 1
  votes[t2] += 1
  votes[t3] -= 1

def get_votes(best_matches, best_dists, Sigma, n_eigenvectors, K):
  '''
  gets edge votes for the triplets

  returns an (n_eigenvectors x n_eigenvectors) numpy array of edges,
  where n_eigenvectors is the number of eigenvectors being examined
  '''

  scores = np.zeros((n_eigenvectors, n_eigenvectors))
  votes = np.zeros(n_eigenvectors)
  for triplet, dist in zip(best_matches, best_dists):
    vote(scores, votes, Sigma, triplet, dist)

  print(votes)
  edges = np.where(votes>K)[0]
  edge_scores = np.zeros((len(edges), len(edges)))
  for i in range(edge_scores.shape[0]):
    for j in range(edge_scores.shape[1]):
      if i == j:
        edge_scores[i][j] = 0
      edge_scores[i][j] = scores[edges[i]][edges[j]]

  edge_scores = np.exp(-edge_scores**2 / 2)
  return edges, edge_scores

def split_eigenvectors(edges, edge_scores):
  '''
  clusters eigenvectors into two separate groups
  '''
  clustering = SpectralClustering(n_clusters=2,  # default: 2
                                  affinity='precomputed',
                                  assign_labels='kmeans',
                                  random_state=0).fit(edge_scores)

  labels = np.zeros((2, len(edges)), dtype='int')
  labels[0,:] = edges
  labels[1,:] = clustering.labels_
  return labels

def main():
  datafile = sys.argv[1]
  with open(datafile) as f:
    params = json.load(f)

  # unpack parameters
  print("\nParameters...")
  print(params)

  name = params['name']
  test_name = params['test_name']
  precomputed = params['precomputed']
  l1 = np.sqrt(np.pi) + params['l1']
  l2 = params['l2']
  noise = params['noise']
  n_samples = params['n_samples']
  seed = params['seed']
  datatype = params['datatype']
  sigma = params['sigma']
  n_comps = params['n_comps']
  n_eigenvectors = params['n_eigenvectors']
  lambda_thresh = params['lambda_thresh']
  K = params['K']
  dist_thresh = params['dist_thresh']

  # filenames
  data_filename = './data/data_' + name + '.dat'  # switch to path containing data
  phi_filename = './data/phi_' + name + '.dat'
  Sigma_filename = './data/Sigma_' + name + '.dat'
  matches_filename = './data/matches_' + name + '.dat'
  dists_filename = './data/dists_' + name + '.dat'

  if precomputed:
    # load data
    print("\nLoading data...")
    data = np.loadtxt(data_filename)[1:,:]

    print("\nLoading phi, Sigma...")
    phi = np.loadtxt(phi_filename)
    Sigma = np.loadtxt(Sigma_filename)

    print("\nLoading matches and distances...")
    matches = np.loadtxt(matches_filename)
    dists = np.loadtxt(dists_filename)
  else:
    # generate random data
    print("\nGenerating random data...")
    data = generate_data(l1, l2, noise=noise, n_samples=n_samples, seed=seed, datatype=datatype)
    np.savetxt(data_filename, data)

    # compute eigenvectors
    print("\nComputing eigenvectors...")
    W = calc_W(data, sigma=sigma)
    phi, Sigma = calc_vars(data, W, n_comps=n_comps)

    np.savetxt(phi_filename, phi)
    np.savetxt(Sigma_filename, Sigma)

  # find triplets
  print("\nComputing triplets...")
  matches, dists = find_triplets(phi, Sigma, n_comps, lambda_thresh)
  print(matches)
  print(dists)

  print("\nSigma...")
  print(Sigma)

  matches_list = []
  dists_list = []
  for match in list(matches):
    matches_list.append([matches[match][0], matches[match][1], match])
    dists_list.append(dists[match])

  dist_order = np.argsort(dists_list)
  ordered_matches = [matches_list[i] for i in dist_order]
  print('\n%d matches found\n' % (len(ordered_matches)), ordered_matches)

  # split eigenvectors
  print("\nSplitting eigenvectors...")
  eigenvectors, eigenvector_scores = get_votes(ordered_matches,
                                               dist_order,
                                               Sigma,
                                               n_eigenvectors,
                                               K)
  labels = split_eigenvectors(eigenvectors, eigenvector_scores)

  if labels[1][0] == 0:
    manifold1 = labels[0][np.where(labels[1]==0)[0]]
    manifold2 = labels[0][np.where(labels[1]==1)[0]]
  else:
    manifold2 = labels[0][np.where(labels[1]==0)[0]]
    manifold1 = labels[0][np.where(labels[1]==1)[0]]

  print("Manifold #1: ", manifold1)
  print("Manifold #2: ", manifold2)

  np.savetxt('./data/manifold1_{}_{}.dat'.format(name, test_name), manifold1)
  np.savetxt('./data/manifold2_{}_{}.dat'.format(name, test_name), manifold2)


if __name__ == '__main__':
  main()
