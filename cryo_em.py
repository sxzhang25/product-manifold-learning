import sys
import json
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

from fakekv import *
import synthetic as syn
from plots import *

###
# cryo_em.py
#
# Generates synthetic cryo-EM data.
###

def generate_data(n_samples, x=10, y=10, var=1, seed=0):
  '''
  Generates synthetic cryo-EM data using the FakeKV class. The synthetic
  molecule has two independent conformational components: a rotation and a
  translation.

  n_samples: the number of images to generate
  x: the largest possible translation in either x-direction
  y: the largest possible translation in either y-direction
  var: the variance of gaussian noise added
  seed: the random seed
  '''
  np.random.seed(seed)
  data = np.zeros((n_samples, L, L))
  mol = FakeKV()
  for i in range(n_samples):
    # show progress
    if (i % 100 == 0):
      print(i, end=" ", flush=True)

    angle = 360 * np.random.random()
    shift_x = x * (2 * np.random.random() - 1)
    shift_y = y * (2 * np.random.random() - 1)
    vol = mol.generate(angle=angle, shift=(shift_x, shift_y))
    vol = ndimage.rotate(vol, -30, (0,2), reshape=False, order=1)

    # project vol in direction of z-axis
    projz = np.sum(vol, axis=2, keepdims=False)

    # add gaussian noise
    gauss = np.random.normal(0, var**0.5, (L,L))
    noisy = projz + gauss

    data[i,:,:] = noisy

  return data

def downsample_data(data, l):
  '''
  downsamples data with resolution L * L to resolution l * l
  '''
  stride = L // l
  lowres_data = data[:,::stride,::stride]
  return lowres_data

def plot_data(data):
  '''
  plots the images of the synthetic data
  '''
  n_samples = data.shape[0]
  rows = int(np.ceil(n_samples**0.5))
  cols = rows
  fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
  for r in range(rows):
    for c in range(cols):
      ax = axs[r, c]
      index = r * cols + c
      if index >= n_samples:
        ax.set_visible(False)
      else:
        vol = data[index]
        ax.imshow(vol)

  plt.show()

def main():
  datafile = sys.argv[1]
  with open(datafile) as f:
    params = json.load(f)

  # unpack parameters
  print("\nParameters...")
  for item, amount in params.items():
    print("{:>15}:  {}".format(item, amount))

  test_name = params['test_name']
  precomputed = params['precomputed']
  var = params['var']
  x = params['x']
  y = params['y']
  n_samples = params['n_samples']
  seed = params['seed']
  sigma = params['sigma']
  n_comps = params['n_comps']
  n_eigenvectors = params['n_eigenvectors']
  lambda_thresh = params['lambda_thresh']
  K = params['K']

  generate_plots = True

  if precomputed:
    info = pickle.load(open("./data/{}_info.pickle".format(test_name), "rb"))

    # load data
    print("\nLoading data...")
    data = info['data']

    # downsample data
    print("\nDownsampling data...")
    t0 = time.perf_counter()
    data = downsample_data(data, 16)
    data = np.reshape(data, (data.shape[0], -1))
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    # apply dimensionality reduction to the data
    print("\nApplying PCA and scaling...")
    t0 = time.perf_counter()
    transformer = KernelPCA(n_components=8, kernel='linear')
    data = transformer.fit_transform(data)

    # standard scale each channel
    scaler = StandardScaler()
    data = np.reshape(data, (-1, data.shape[1]))
    data = scaler.fit_transform(data)
    data = np.reshape(data, (n_samples, -1))
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    # load phi, Sigma
    print("\nLoading phi, Sigma...")
    phi = info['phi']
    Sigma = info['Sigma']

    # get triplets
    print("\nLoading triplets...")
    matches = info['matches']
    dists = info['dists']

  else:
    # create a dictionary to store all information in
    info = {}

    # generate random data
    print("\nGenerating random data...")
    data = generate_data(n_samples, x=x, y=y, var=var)
    info['data'] = data

    # downsample data
    print("\nDownsampling data...")
    t0 = time.perf_counter()
    data = downsample_data(data, 16)
    data = np.reshape(data, (data.shape[0], -1))
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    # apply dimensionality reduction to the data
    print("\nApplying PCA and scaling...")
    t0 = time.perf_counter()
    transformer = KernelPCA(n_components=8, kernel='linear')
    data = transformer.fit_transform(data)

    # standard scale each channel
    scaler = StandardScaler()
    data = np.reshape(data, (-1, data.shape[1]))
    data = scaler.fit_transform(data)
    data = np.reshape(data, (n_samples, -1))
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    # compute eigenvectors
    print("\nComputing eigenvectors...")
    t0 = time.perf_counter()
    W = syn.calc_W(data, sigma)
    phi, Sigma = syn.calc_vars(data, W, sigma, n_comps=n_comps)
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    info['phi'] = phi
    info['Sigma'] = Sigma

    # find triplets
    print("\nComputing triplets...")
    t0 = time.perf_counter()
    matches, dists = syn.find_triplets(phi, Sigma, n_comps, lambda_thresh)
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    info['matches'] = matches
    info['dists'] = dists

  # split eigenvectors
  print("\nSplitting eigenvectors...")
  t0 = time.perf_counter()
  labels = syn.split_eigenvectors(matches,
                                  dists,
                                  n_eigenvectors,
                                  K,
                                  n_clusters=2)
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
  print("Saving data...")
  with open('./data/{}_info.pickle'.format(test_name), 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

  # plot eigenvectors
  vecs = [phi[:,i] for i in range(n_eigenvectors)]
  eigenvectors_filename = './images/' + test_name + '_eigenvalues_' + str(n_eigenvectors) + '.png'
  plot_eigenvectors(data[:,:3],
                             vecs[:20],
                             labels=[int(i) for i in range(n_eigenvectors)],
                             title='Laplace Eigenvectors',
                             filename=eigenvectors_filename)

  vecs1 = [phi[:,int(i)] for i in manifold1]
  vecs2 = [phi[:,int(i)] for i in manifold2]

  plot_eigenvectors(data,
                    vecs1,
                    labels=[int(i) for i in manifold1],
                    filename='./images/manifold1_{}.png'.format(test_name))

  plot_eigenvectors(data,
                    vecs2,
                    labels=[int(i) for i in manifold1],
                    filename='./images/manifold2_{}.png'.format(test_name))

if __name__ == '__main__':
  main()
