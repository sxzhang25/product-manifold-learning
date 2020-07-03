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
    data = np.row_stack(([l1, l2], data))

  elif datatype=='line_circle':
    # line segment and circle
    line_data = l1 * (np.random.rand(n_samples) + noise * (2 * np.random.rand(n_samples) - 1))
    circle_data = np.empty((n_samples,2))
    for i in range(n_samples):
      theta = 2 * np.pi * np.random.rand()
      circle_data[i,:] = [(l2 + noise * (2 * np.random.rand(n_samples) - 1)) * np.cos(theta), l2 * np.sin(theta)]
    data = np.column_stack((line_data, circle_data))
    data = np.row_stack(([l1, l2, 0], data))

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
    data = np.row_stack(([l1, l2, 0, 0], data))

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
    data = np.row_stack(([l1, l2, 0, 0], data))

  else:
    print('Error: invalid data type')
    return

  return data


###
# COMPUTE EIGENVECTORS
###

def calc_W(data, sigma=None):
  '''
  calculates the weight matrix W
  '''

  t0 = time.perf_counter()
  pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))

  if sigma is None:
    sigma = 0
    for i in range(data.shape[0]):
      sigma += np.min(np.delete(pairwise_sq_dists[i,:], i))
    sigma /= data.shape[0]
    # sigma = 1

  W = np.exp(-pairwise_sq_dists / sigma)
  t1 = time.perf_counter()
  print("  Calculating W took %2.2f seconds" % (t1-t0))

  return W

def calc_vars(data, W, n_comps=100):
  '''
  calculates phi, psi, and Sigma
  '''
  # ones = np.ones(W.shape[0])
  # p = W @ ones
  # W2 = W / np.outer(p, p)
  # v = np.sqrt(W2 @ ones)
  # S = W2 / np.outer(v, v)
  #
  # V, Sigma, VT = randomized_svd(S,
  #                               n_components=n_comps+1,
  #                               n_iter=5,
  #                               random_state=None)
  # phi = V / V[:,0][None].T

  # ones = np.ones(W.shape[0])
  # v = np.sqrt(W @ ones)
  # S = W / np.outer(v, v)
  #
  # V, Sigma, VT = randomized_svd(S,
  #                               n_components=n_comps+1,
  #                               n_iter=5,
  #                               random_state=None)
  # phi = V / V[:,0][None].T


  # calculate D, M, S (use coifmann-lafon diffusion map)
  t0 = time.perf_counter()
  d = np.zeros(data.shape[0])
  for i in range(d.shape[0]):
    d[i] = np.sum(W[i,:])
  D = np.diag(d)
  Dinv = scipy.linalg.inv(D)

  # double-normalize W and recompute degree matrix
  W_ = Dinv @ W @ Dinv
  d = np.zeros(data.shape[0])
  for i in range(d.shape[0]):
    d[i] = np.sum(W_[i,:])
  D_ = np.diag(d)
  D_inv = scipy.linalg.inv(D_)
  D_sqrt = np.sqrt(D_)
  D_sqrt_inv = scipy.linalg.inv(D_sqrt)

  # compute eigenvectors using SVD of symmetric matrix S = D^-1 M D^-1
  M = D_inv @ W_
  S = D_sqrt @ M @ D_sqrt_inv
  t1 = time.perf_counter()
  print("  Calculating variables took %2.2f seconds" % (t1-t0))

  # use svd
  t0 = time.perf_counter()
  V, Sigma, VT = randomized_svd(S,
                                n_components=n_comps+1,
                                n_iter=5,
                                random_state=None)

  phi = D_sqrt_inv @ V
  t1 = time.perf_counter()
  print("  Calculating eigenvectors took %2.2f seconds" % (t1-t0))

  # compute approximate eigenvalues of Laplacian
  L = D - W
  for i in range(phi.shape[1]):
    Sigma[i] = np.abs(np.average((L @ phi[:,i]) / phi[:,i]))

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
  score = np.linalg.norm((vs1 - vs2), ord=1) / vs1.shape[0]
  return score

def find_match(data, v, a, candidates, Sigma, eps=10e-1):
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

  return best_match, best_dist

# def find_matches(data, phi, Sigma, n_eigenvectors=100):
#   '''
#   find all best triplets in the first n eigenvectors of the data
#   (excluding 0 eigenvector)
#   '''
#
#   t0 = time.perf_counter()
#   matches = np.zeros((n_eigenvectors, n_eigenvectors), dtype=int)
#   dists = np.zeros((n_eigenvectors, n_eigenvectors))
#   for i in range(1, n_eigenvectors+1):
#     for j in range(i+1, n_eigenvectors+1):
#       v1 = phi[:,i]
#       v2 = phi[:,j]
#       v = v1 * v2
#       a = Sigma[i] + Sigma[j]
#       match, dist = find_match(data, v, a, phi[:,j+1:], Sigma[j+1:])
#       matches[i-1,j-1] = match + j + 1
#       matches[j-1,i-1] = match + j + 1
#       dists[i-1,j-1] = dist
#       dists[j-1,i-1] = dist
#
#   t1 = time.perf_counter()
#   print("Finding best matches took %2.2f seconds" % (t1-t0))
#   return matches, dists
#
# def find_best_matches(phi, matches, dists, dist_thresh):
#   '''
#   returns a list of all matches satisfy the threshold
#   '''
#
#   best_matches = {}
#   best_dists = {}
#   for i in range(dists.shape[0]):
#     for j in range(i+1, dists.shape[0]):
#       if dists[i,j] < dist_thresh:
#         v1, v2, match = [i+1, j+1, matches[i,j]]
#         if match not in best_matches:
#           best_matches[match] = [v1, v2, match]
#           best_dists[match] = dists[i,j]
#         else:
#           if dists[i,j] < best_dists[match]:
#             best_matches[match] = [v1, v2, match]
#             best_dists[match] = dists[i,j]
#
#   return best_matches, best_dists

def find_best_matches(data, phi, Sigma, n_eigenvectors=100, eps=10e-2):
  best_matches = {}
  best_dists = {}
  for i in range(1, n_eigenvectors+1):
    for j in range(i+1, n_eigenvectors+1):
      v1 = phi[:,i]
      v2 = phi[:,j]
      v = v1 * v2
      a = Sigma[i] + Sigma[j]
      match, dist = find_match(data, v, a, phi[:,j+1:], Sigma[j+1:], eps=eps)
      triplet = [i, j, match + j + 1]
      best_matches[match] = triplet
      best_dists[match] = dist

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
    data = data[1:,:]

    # compute eigenvectors
    print("\nComputing eigenvectors...")
    W = calc_W(data, sigma=sigma)
    phi, Sigma = calc_vars(data, W, n_comps=n_comps)

    np.savetxt(phi_filename, phi)
    np.savetxt(Sigma_filename, Sigma)

  # find triplets
  print("\nComputing triplets...")
  matches, dists = find_best_matches(data, phi, Sigma, n_eigenvectors)

  # np.savetxt(matches_filename, matches)
  # np.savetxt(dists_filename, dists)

  # # find best matches
  # print("\nFinding best matches...")
  # best_matches, best_dists = find_best_matches(phi, matches, dists, dist_thresh)

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
