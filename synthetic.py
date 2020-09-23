import time
import sys
import json
import pickle
from itertools import combinations

import numpy as np
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import SpectralClustering
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cp

from plots import *

###
# synthetic.py
#
# Generate synthetic random data sampled from a toy product manifold and compute
# the Laplacian, eigenvalues, and independent manifolds of the data
###

def generate_data(dimensions, n_samples, datatype='rectangle', seed=0, noise=0.05):
  '''
  generates uniform random data
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
    circle1_data[:,0] = (r1 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle1_data[:,1] = (r1 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    circle2_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle2_data[:,0] = (r2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle2_data[:,1] = (r2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    data = np.column_stack((circle1_data, circle2_data))

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

def calc_vars(data, W, sigma, n_eigenvectors):
  '''
  calculates phi and Sigma
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

def calculate_corr(v_i, v_j):
  '''
  calculates proximity of v_i to v_j
  '''

  # normalize vectors to unit norm
  v_i /= np.linalg.norm(v_i)
  v_j /= np.linalg.norm(v_j)

  # calculate L2 distance between v1 and v2
  corr = np.dot(v_i, v_j)
  return corr # score

def find_combos(phi, Sigma, n_comps=2, lambda_thresh=10e-3, corr_thresh=0.5):
  best_matches = {}
  max_corrs = {}
  all_corrs = {}

  for k in range(n_comps, phi.shape[1]):
    # show progress
    if (k % 10 == 0):
      sys.stdout.write('\r')
      sys.stdout.write("[%-20s] %d%%" % ('='*int(20*k/phi.shape[1]), 100*k/phi.shape[1]))
      sys.stdout.flush()

    v_k = phi[:,k]
    lambda_k = Sigma[k]
    max_corr = 0
    best_match = []

    for m in range(1, n_comps + 1):
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

def split_eigenvectors(best_matches, best_corrs, n_eigenvectors, n_comps=2):
  '''
  clusters eigenvectors into two separate groups
  '''
  votes = np.zeros(n_eigenvectors)
  W = np.zeros((n_eigenvectors, n_eigenvectors))

  print("\nCombos...")
  for match in list(best_matches):
    combo = best_matches[match]
    print(combo, match)
    for pair in list(combinations(combo, 2)):
      W[pair[0]][pair[1]] += best_corrs[match]
      W[pair[1]][pair[0]] += best_corrs[match]
      votes[pair[0]] += best_corrs[match]
      votes[pair[1]] += best_corrs[match]
  print("\nVotes:\n", votes)

  # perform spectral clustering based on independent vectors
  independent = np.where(votes>0)[0]
  n = len(independent)
  W_ = np.zeros((n, n))
  for i in range(W_.shape[0]):
    for j in range(W_.shape[1]):
      if i == j:
        W_[i][j] = 0
      else:
        W_[i][j] = W[independent[i]][independent[j]]

  np.random.seed(1)
  if n_comps == 2:
    Y = cp.Variable((n, n), PSD=True)
    constraints = [cp.diag(Y) == 1]
    obj = 0.5 * cp.sum(cp.multiply(W_, np.ones((n, n)) - Y))
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve()

    eigenvalues, eigenvectors = np.linalg.eigh(Y.value)
    eigenvalues = np.maximum(eigenvalues, 0)
    diagonal_root = np.diag(np.sqrt(eigenvalues))
    assignment = diagonal_root @ eigenvectors.T
    partition = np.random.normal(size=n)
    projections = assignment.T @ partition

    labels = np.zeros((2, len(independent)), dtype='int')
    labels[0,:] = independent
    for i in range(n):
      labels[1][i] = 1 if projections[i] >= 0 else 0

  elif n_comps > 2:
    # follow heuristic according to
    # https://drops.dagstuhl.de/opus/volltexte/2018/8309/pdf/OASIcs-SOSA-2018-13.pdf

    Y = cp.Variable((n,n), PSD=True)
    constraints = [cp.diag(Y) == 1]
    for i in range(n):
      for j in range(n):
        if j is not i:
          constraints += [Y[i,j] >= -1 / (n_comps - 1)]

    obj = (1 - 1 / n_comps) * cp.sum(cp.multiply(W_, np.ones((n, n)) - Y))
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
    for i in range(n):
      theta_old = theta[i]
      if g_assignment[i] >= 0:
        if g_assignment_[i] < 0:
          print('case 1')
          theta[i] += 2 * np.pi
        else:
          print('case 0')
      else:
        print('case 3')
        theta[i] += np.pi
      print(theta_old, theta[i])

    z = 2 * np.pi * np.random.random()
    print(z)
    plt.figure()
    for i in theta:
      plt.plot([0, np.cos(i)], [0, np.sin(i)], c='red')
    for j in range(n_comps):
      z_angle = (z + j * 2 * np.pi / n_comps) % (2 * np.pi)
      plt.plot([0, np.cos(z_angle)], [0, np.sin(z_angle)], c='black')
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.axis('equal')
    plt.show()

    # partition = np.random.normal(size=(n, n_comps))
    # projections = assignment.T @ partition

    labels = np.zeros((2, len(independent)), dtype='int')
    labels[0,:] = independent
    for i in range(n):
      for j in range(n_comps):
        if (z + j * 2 * np.pi / n_comps) % (2 * np.pi) <= theta[i] and theta[i] < (z + j * 2 * np.pi / n_comps) % (2 * np.pi) + j * 2 * np.pi / n_comps:
          labels[1][i] = j

  return labels

  # W_ = np.exp(-W_**2)
  # np.set_printoptions(precision=3)
  # clustering = SpectralClustering(n_clusters=n_comps,
  #                                 affinity='precomputed',
  #                                 assign_labels='kmeans',
  #                                 random_state=0).fit(W_)
  #
  # labels = np.zeros((2, len(independent)), dtype='int')
  # labels[0,:] = independent
  # labels[1,:] = clustering.labels_
  # return labels

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
  dimensions = params['dimensions']
  dimensions[0] = np.sqrt(np.pi) + dimensions[0]
  noise = params['noise']
  n_samples = params['n_samples']
  seed = params['seed']
  datatype = params['datatype']
  sigma = params['sigma']
  n_comps = params['n_comps']
  n_eigenvectors = params['n_eigenvectors']
  lambda_thresh = params['lambda_thresh']
  corr_thresh = params['corr_thresh']

  generate_plots = False

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
    best_matches = info['best_matches']
    best_corrs = info['best_corrs']
    all_corrs = info['all_corrs']

    if datatype == 'line_circle':
      data_r = np.zeros((data.shape[0],2))
      data_r[:,0] = data[:,0]
      data_r[:,1] = np.arctan2(data[:,2], data[:,1])
    else:
      data_r = data
  else:
    # create a dictionary to store all information in
    info = {}

    # generate random data
    print("\nGenerating random data...")
    data = generate_data(dimensions,
                         noise=noise,
                         n_samples=n_samples,
                         datatype=datatype,
                         seed=seed)
    info['data'] = data

    if datatype == 'line_circle':
      data_r = np.zeros((data.shape[0],2))
      data_r[:,0] = data[:,0]
      data_r[:,1] = np.arctan2(data[:,2], data[:,1])
    else:
      data_r = data

    # compute eigenvectors
    print("\nComputing eigenvectors...")
    t0 = time.perf_counter()
    W = calc_W(data_r, sigma)
    phi, Sigma = calc_vars(data_r, W, sigma, n_eigenvectors=n_eigenvectors)
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    info['phi'] = phi
    info['Sigma'] = Sigma

  # find combos
  print("\nComputing combos...")
  t0 = time.perf_counter()
  best_matches, best_corrs, all_corrs = find_combos(phi, Sigma, n_comps, lambda_thresh, corr_thresh)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))
  info['best_matches'] = best_matches
  info['best_corrs'] = best_corrs
  info['all_corrs'] = all_corrs

  # split eigenvectors
  print("\nSplitting eigenvectors...")
  t0 = time.perf_counter()
  labels = split_eigenvectors(best_matches, best_corrs, n_eigenvectors, n_comps=n_comps)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  manifolds = []
  for m in range(n_comps):
    manifold = labels[0][np.where(labels[1]==m)[0]]
    manifolds.append(manifold)
    print("Manifold #{}".format(m), manifold)

  info['manifolds'] = manifolds

  # save info dictionary using pickle
  with open('./data/{}_info.pickle'.format(test_name), 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)

  ###
  # GENERATE PLOTS
  ###

  for i in range(len(manifolds)):
    plot_embedding(phi, manifolds[i][:min(2, len(manifolds[0]))])
  plot_correlations(all_corrs, thresh=corr_thresh)

  if generate_plots:
    # vecs = [phi[:,i] for i in range(n_eigenvectors)]
    # plot_eigenvectors(data_r[:,:2],
    #                   vecs[:25],
    #                   labels=[int(i) for i in range(n_eigenvectors)],
    #                   title='Laplace Eigenvectors',
    #                   filename='./images/' + test_name + '_eigenvalues_' + '_(0,1)' + '.png')
    # plot_eigenvectors(data_r[:,1:],
    #                   vecs[:25],
    #                   labels=[int(i) for i in range(n_eigenvectors)],
    #                   title='Laplace Eigenvectors',
    #                   filename='./images/' + test_name + '_eigenvalues_' + '_(1,2)' + '.png')
    # plot_eigenvectors(data_r[:,[0,2]],
    #                   vecs[:25],
    #                   labels=[int(i) for i in range(n_eigenvectors)],
    #                   title='Laplace Eigenvectors',
    #                   filename='./images/' + test_name + '_eigenvalues_' + '_(0,2)' + '.png')
    if not precomputed:
      # plot original data
      plot_data(data,
                title='Original Data',
                filename='./images/{}_original_data.png'.format(test_name))

      # plot eigenvectors
      vecs = [phi[:,i] for i in range(n_eigenvectors)]
      eigenvectors_filename = './images/' + test_name + '_eigenvalues_' + str(n_eigenvectors) + '.png'
      plot_eigenvectors(data_r,
                        vecs[:25],
                        labels=[int(i) for i in range(n_eigenvectors)],
                        title='Laplace Eigenvectors',
                        filename=eigenvectors_filename)

    # plot best matches
    manifolds = info['manifolds']

    independent_vecs = []
    for manifold in manifolds:
      vecs = [phi[:,int(i)] for i in manifold]
      independent_vecs.append(vecs)

    for i,vecs in enumerate(independent_vecs):
      plot_eigenvectors(data_r,
                        vecs,
                        labels=[int(j) for j in manifolds[i]],
                        filename='./images/manifold{}_{}.png'.format(i, test_name))

if __name__ == '__main__':
  main()
