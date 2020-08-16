import time
import sys
import json
import pickle

import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import SpectralClustering
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plots import *

###
# synthetic.py
#
# Generate synthetic random data sampled from a toy product manifold and compute
# the Laplacian, eigenvalues, and independent manifolds of the data
###

def generate_data(l1, l2, n_samples, datatype='line_line', seed=0, noise=0.05):
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
    line1_data = l1 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line2_data = l2 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    data = np.column_stack((line1_data, line2_data))

  elif datatype=='line_circle':
    # line segment and circle
    line_data = l1 * (np.random.rand(n_samples) +  \
                      np.random.normal(scale=noise, size=n_samples))
    circle_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle_data[:,0] = (l2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle_data[:,1] = (l2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    data = np.column_stack((line_data, circle_data))

  elif datatype=='circle_circle':
    # circle and circle
    circleA_data = np.empty((n_samples,2))
    circleB_data = np.empty((n_samples,2))
    for i in range(n_samples):
      theta = 2 * np.pi * np.random.rand(scale=noise)
      circleA_data[i,:] = [(l1 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta),
                           l1 * np.sin(theta)]
      theta = 2 * np.pi * np.random.normal(scale=noise)
      circleB_data[i,:] = [(l2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta),
                           l2 * np.sin(theta)]
    data = np.column_stack((circleA_data, circleB_data))

  elif datatype=='rect_circle':
    # rectangle and circle
    line1_data = l1 * (np.random.rand(n_samples) + \
                       noise * np.random.normal(scale=noise, size=n_samples))
    line2_data = l2 * (np.random.rand(n_samples) + \
                       noise * np.random.normal(scale=noise, size=n_samples))
    rect_data = np.column_stack((line1_data, line2_data))

    circle_data = np.empty((n_samples,2))
    for i in range(n_samples):
      theta = 2 * np.pi * np.random.rand(scale=noise)
      circle_data[i,:] = [(l1 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta),
                          l1 * np.sin(theta)]
    data = np.column_stack((rect_data, circle_data))

  else:
    print('Error: invalid data type')
    return

  return data

def plot_data(data, title=None, filename=None):
  '''
  plot the original data
  '''

  fig = plt.figure(figsize=(5,5))
  if data.shape[1] <= 2:
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=5)
  elif data.shape[1] == 3:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], s=5)
  else:
    ax = fig.add_subplot(111, projection='3d')
    g = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], s=5)
    cb = plt.colorbar(g)
  if title:
    plt.title(title, pad=10)
  if filename:
    plt.savefig(filename)
  plt.show()

###
# COMPUTE EIGENVECTORS
###

def calc_W(data, sigma):
  '''
  calculates the weight matrix W
  '''

  pairwise_sq_dists = squareform(pdist(data, 'sqeuclidean'))
  W = np.exp(-pairwise_sq_dists / sigma)
  return W

def calc_vars(data, W, sigma, n_comps=100):
  '''
  calculates phi, psi, and Sigma
  '''

  ones = np.ones(W.shape[0])
  v = np.sqrt(W @ ones)
  S = W / np.outer(v, v)
  V, Sigma, VT = randomized_svd(S,
                                n_components=n_comps,
                                n_iter=5,
                                random_state=None)
  phi = V / V[:,0][:,None]
  Sigma = -np.log(Sigma) / sigma
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
    for i in range(1, k):
      for j in range(i+1, k):
        v_i = phi[:,i]
        v_j = phi[:,j]
        v_ij = v_i * v_j
        lambda_ij = Sigma[i] + Sigma[j]
        eig_d = abs(lambda_k - lambda_ij)
        if eig_d < lambda_thresh:
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

  return best_matches, min_dists

###
# VOTING SCHEME
###

def split_eigenvectors(best_matches, dists, n_eigenvectors, K, n_clusters=2):
  '''
  clusters eigenvectors into two separate groups
  '''
  # sort triplets from smallest distance to largest distance (quality of triplet)
  triplets_list = []
  sorted_dists = []
  for match in list(best_matches):
    triplets_list.append([best_matches[match][0], best_matches[match][1], match])
    sorted_dists.append(dists[match])

  print("\nTriplets...")
  for triplet in triplets_list:
    print(triplet)

  votes = np.zeros(n_eigenvectors)
  mixtures = set() # oon't do suppression here (use simpler voting scheme)

  W = np.zeros((n_eigenvectors, n_eigenvectors))
  for triplet in triplets_list:
    v_i, v_j, v_k = triplet
    W[v_i][v_j] += 1
    W[v_j][v_i] += 1
    votes[v_i] += 1
    votes[v_j] += 1

  print("\nVotes:\n", votes)

  # perform spectral clustering based on independent vectors
  independent = np.where(votes>=K)[0]
  W_ = np.zeros((len(independent), len(independent)))
  for i in range(W_.shape[0]):
    for j in range(W_.shape[1]):
      if i == j:
        W_[i][j] = 0
      else:
        W_[i][j] = W[independent[i]][independent[j]]

  W_ = np.exp(-W_**2)
  np.set_printoptions(precision=3)
  clustering = SpectralClustering(n_clusters=n_clusters,  # default: 2
                                  affinity='precomputed',
                                  assign_labels='kmeans',
                                  random_state=0).fit(W_)

  labels = np.zeros((2, len(independent)), dtype='int')
  labels[0,:] = independent
  labels[1,:] = clustering.labels_
  return labels

def main():
  ###
  # GET PARAMETERS
  ###

  datafile = sys.argv[1]
  with open(datafile) as f:
    params = json.load(f)

  # unpack parameters
  print("\nParameters...")
  for item, amount in params.items():
    print("{:>15}:  {}".format(item, amount))

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

  generate_plots = True

  ###
  # DATA GENERATION
  ###

  if precomputed:
    info = pickle.load(open("./data/{}_info.pickle".format(test_name), "rb"))

    # load data
    print("\nLoading data...")
    data = info['data']

    print("\nLoading phi, Sigma...")
    phi = info['phi']
    Sigma = info['Sigma']

    print("\nLoading matches and distances...")
    matches = info['matches']
    dists = info['dists']
  else:
    # create a dictionary to store all information in
    info = {}

    # generate random data
    print("\nGenerating random data...")
    data = generate_data(l1,
                         l2,
                         noise=noise,
                         n_samples=n_samples,
                         datatype=datatype,
                         seed=seed)
    info['data'] = data

    # compute eigenvectors
    print("\nComputing eigenvectors...")
    t0 = time.perf_counter()
    W = calc_W(data, sigma)
    phi, Sigma = calc_vars(data, W, sigma, n_comps=n_comps)
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    info['phi'] = phi
    info['Sigma'] = Sigma

  # find triplets
  print("\nComputing triplets...")
  t0 = time.perf_counter()
  matches, dists = find_triplets(phi, Sigma, n_comps, lambda_thresh)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))
  info['matches'] = matches
  info['dists'] = dists

  # split eigenvectors
  print("\nSplitting eigenvectors...")
  t0 = time.perf_counter()
  labels = split_eigenvectors(matches, dists, n_eigenvectors, K)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  if labels[1][0] == 0:
    manifold1 = labels[0][np.where(labels[1]==0)[0]]
    manifold2 = labels[0][np.where(labels[1]==1)[0]]
  else:
    manifold2 = labels[0][np.where(labels[1]==0)[0]]
    manifold1 = labels[0][np.where(labels[1]==1)[0]]

  print("Manifold #1: ", manifold1)
  print("Manifold #2: ", manifold2)

  info['manifold1'] = manifold1
  info['manifold2'] = manifold2

  # save info dictionary using pickle
  with open('./data/{}_info.pickle'.format(test_name), 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)

  ###
  # GENERATE PLOTS
  ###

  if generate_plots:
    if datatype == 'line_circle':
      data_r = np.zeros((data.shape[0],2))
      data_r[:,0] = data[:,0]
      data_r[:,1] = np.arctan2(data[:,2], data[:,1])
    elif datatype == 'rect_circle':
      data_r = np.zeros((data.shape[0],3))
      data_r[:,0] = data[:,0]
      data_r[:,1] = data[:,1]
      data_r[:,2] = np.arctan2(data[:,3], data[:,2])
    else:
      data_r = data

    if not precomputed:
      # plot original data
      plot_data(data,
                title='Original Data',
                filename='./images/{}_original_data.png'.format(test_name))

      # plot eigenvectors
      vecs = [phi[:,i] for i in range(n_eigenvectors)]
      eigenvectors_filename = './images/' + test_name + '_eigenvalues_' + str(n_eigenvectors) + '.png'
      plot_eigenvectors(data_r,
                        vecs[:100],
                        labels=[int(i) for i in range(n_eigenvectors)],
                        title='Laplace Eigenvectors',
                        filename=eigenvectors_filename)

    # plot best matches
    manifold1 = info['manifold1']
    manifold2 = info['manifold2']

    vecs1 = [phi[:,int(i)] for i in manifold1]
    vecs2 = [phi[:,int(i)] for i in manifold2]

    plot_eigenvectors(data_r,
                      vecs1,
                      labels=[int(i) for i in manifold1],
                      filename='./images/manifold1_{}.png'.format(test_name))

    plot_eigenvectors(data_r,
                      vecs2,
                      labels=[int(i) for i in manifold2],
                      filename='./images/manifold2_{}.png'.format(test_name))

    plot_independent_eigenvectors(manifold1,
                                  manifold2,
                                  n_eigenvectors,
                                  title='manifold split',
                                  filename='./images/{}_{}_eigenvector_division.png'.format(test_name, K))

if __name__ == '__main__':
  main()
