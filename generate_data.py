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
                                n_components=n_comps+1,
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

def scale(v1, v2):
  '''
  scales v1 to match the range of v2
  '''

  vs = (np.max(v2) - np.min(v2))/(np.max(v1) - np.min(v1)) * (v1 - np.min(v1)) + np.min(v2)
  return vs

def calculate_score(data, v1, v2):
  '''
  calculates proximity of v2 to v1
  '''

  # scale v1, v2 to [-1,1]
  vs1 = scale(v1, [-1,1])
  vs2 = scale(v2, [-1,1])

  # calculate L1 distance between v1 and v2
  score = np.linalg.norm((vs1 - vs2), ord=2) / vs1.shape[0]
  return score

def find_match(data, v, a, candidates, Sigma, eps=1.5):
  '''
  finds the best match to v out of candidates, according to score
  returns the best match and its distance from v

  v: the eigenvector (product of two base vectors) to match
  a: the eigenvalue to match (sum of two base eigenvalues)
  candidates: an array of vectors (vertically concatenated)
  Sigma: the eigenvalues of the data
  '''

  best_match = 0
  best_dist = np.inf
  for i in range(candidates.shape[1]):
    if abs(a - Sigma[i]) / a < eps:
      u = candidates[:,i]

      # test with positive
      d = calculate_score(data, v, u)
      if d < best_dist:
        best_match = i
        best_dist = d

      # test with negative
      d = calculate_score(data, v, -u)
      if d < best_dist:
        best_match = i
        best_dist = d

  # print(best_match, best_dist)
  return best_match, best_dist

def find_best_matches(data, phi, Sigma, dist_thresh, n_eigenvectors=100, eps=10e-3):
  best_matches = {}
  best_dists = {}
  for i in range(1, n_eigenvectors+1):
    for j in range(i+1, n_eigenvectors+1):
      v1 = phi[:,i]
      v2 = phi[:,j]
      v = v1 * v2
      a = Sigma[i] + Sigma[j]
      match, dist = find_match(data, v, a, phi[:,j+1:], Sigma[j+1:], eps=eps)
      if dist < dist_thresh:
        triplet = (i, j, match + j + 1)
        if triplet[2] not in best_dists:
          best_matches[triplet[2]] = triplet
          best_dists[triplet[2]] = dist
        else:
          if dist < best_dists[triplet[2]]:
            best_matches[triplet[2]] = triplet
            best_dists[triplet[2]] = dist

  return best_matches, best_dists


###
# VOTING SCHEME
###
def vote(edge_scores, votes, triplet, dist):
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
  edge_scores[t1][t2] += np.exp(dist)
  edge_scores[t2][t1] += np.exp(dist)
  votes[t1] += 1
  votes[t2] += 1
  votes[t3] -= 1

def get_votes(best_matches, best_dists, n_eigenvectors, K):
  '''
  gets edge votes for the triplets

  returns an (n_eigenvectors x n_eigenvectors) numpy array of edges,
  where n_eigenvectors is the number of eigenvectors being examined
  '''

  scores = np.zeros((n_eigenvectors, n_eigenvectors))
  votes = np.zeros(n_eigenvectors, dtype='int')
  for triplet, dist in zip(best_matches, best_dists):
    if (triplet[0] < n_eigenvectors and
      triplet[1] < n_eigenvectors and
      triplet[2] < n_eigenvectors):
      vote(scores, votes, triplet, dist)

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
  eps = params['eps']
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
  matches, dists = find_best_matches(data, phi, Sigma, dist_thresh, n_eigenvectors, eps)
  print(matches)
  print(dists)

  print("\nSigma...")
  print(Sigma)

  matches_list = []
  dists_list = []
  for match in list(matches):
    matches_list.append(matches[match])
    dists_list.append(dists[match])

  dist_order = np.argsort(dists_list)
  ordered_matches = [matches_list[i] for i in dist_order]
  print('\n%d matches found\n' % (len(ordered_matches)), ordered_matches)

  # split eigenvectors
  print("\nSplitting eigenvectors...")
  eigenvectors, eigenvector_scores = get_votes(ordered_matches,
                                               dist_order,
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
