import sys
import json
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from sklearn.decomposition import PCA
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
  image_data = np.zeros((n_samples, L, L))
  raw_data = np.zeros((n_samples, 3))
  mol = FakeKV()
  for i in range(n_samples):
    # show progress
    if (i % 10 == 0):
      sys.stdout.write('\r')
      sys.stdout.write("[%-20s] %d%%" % ('='*int(20*i/n_samples), 100*i/n_samples))
      sys.stdout.flush()

    angle = 90 * np.random.random()
    shift_x = x * (2 * np.random.random() - 1)
    shift_y = y * (2 * np.random.random() - 1)
    vol = mol.generate(angle=angle, shift=(shift_x, shift_y))
    vol = ndimage.rotate(vol, -30, (0,2), reshape=False, order=1)

    # project vol in direction of z-axis
    projz = np.sum(vol, axis=2, keepdims=False)

    # add gaussian noise
    gauss = np.random.normal(0, var**0.5, (L,L))
    noisy = projz + gauss

    image_data[i,:,:] = noisy
    raw_data[i,:] = [shift_x, shift_y, angle]

  return image_data, raw_data

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
  corr_thresh = params['corr_thresh']

  generate_plots = False

  if precomputed:
    info = pickle.load(open("./data/{}_info.pickle".format(test_name), "rb"))

    # load data
    print("\nLoading data...")
    image_data = info['image_data']
    raw_data = info['raw_data']
    # plot_data(image_data[:9])

  else:
    # create a dictionary to store all information in
    info = {}

    # generate random data
    print("\nGenerating random data...")
    image_data, raw_data = generate_data(n_samples, x=x, y=y, var=var)
    info['image_data'] = image_data
    info['raw_data'] = raw_data

  # apply PCA and standard scaling to the data
  print("\nApplying PCA and standard scaling...")
  t0 = time.perf_counter()
  image_data = np.reshape(image_data, (n_samples, -1))
  pca = PCA(n_components=4)
  image_data = pca.fit_transform(image_data)

  # standard scale each channel
  scaler = StandardScaler()
  image_data = np.reshape(image_data, (-1, image_data.shape[1]))
  image_data = scaler.fit_transform(image_data)
  image_data = np.reshape(image_data, (n_samples, -1))
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  # compute eigenvectors
  print("\nComputing eigenvectors...")
  t0 = time.perf_counter()
  W = syn.calc_W(image_data, sigma)
  phi, Sigma = syn.calc_vars(image_data, W, sigma, n_eigenvectors=n_eigenvectors)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  info['phi'] = phi
  info['Sigma'] = Sigma

  # find combos
  print("\nComputing combos...")
  t0 = time.perf_counter()
  best_matches, best_corrs, all_corrs = syn.find_combos(phi, Sigma, n_comps, lambda_thresh, corr_thresh)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  info['best_matches'] = best_matches
  info['best_corrs'] = best_corrs
  info['all_corrs'] = all_corrs

  # split eigenvectors
  print("\nSplitting eigenvectors...")
  t0 = time.perf_counter()
  labels = syn.split_eigenvectors(best_matches, best_corrs, n_eigenvectors, n_comps=n_comps)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  manifolds = []
  for m in range(n_comps):
    manifold = labels[0][np.where(labels[1]==m)[0]]
    manifolds.append(manifold)
    print("Manifold #{}".format(m), manifold)

  info['manifolds'] = manifolds

  # save info dictionary using pickle
  print("Saving data...")
  with open('./data/{}_info.pickle'.format(test_name), 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

  for i in range(len(manifolds)):
    plot_embedding(phi, manifolds[i][:min(2, len(manifolds[0]))])
  plot_correlations(all_corrs, thresh=corr_thresh)

  # plot eigenvectors
  if generate_plots:
    vecs = [phi[:,i] for i in range(n_eigenvectors)]
    eigenvectors_filename = './images/' + test_name + '_eigenvalues.png'

    plot_eigenvectors(raw_data,
                      vecs[:25],
                      labels=[int(i) for i in range(n_eigenvectors)],
                      title='Laplace Eigenvectors',
                      filename=eigenvectors_filename )

    # plot best matches
    manifolds = info['manifolds']

    independent_vecs = []
    for manifold in manifolds:
      vecs = [phi[:,int(i)] for i in manifold]
      independent_vecs.append(vecs)

    for i,vecs in enumerate(independent_vecs):
      plot_eigenvectors(raw_data,
                        vecs,
                        labels=[int(j) for j in manifolds[i]],
                        filename='./images/manifold{}_{}.png'.format(i, test_name))
      plot_eigenvectors(raw_data[:,[0,2]],
                        vecs,
                        labels=[int(j) for j in manifolds[i]],
                        filename='./images/manifold{}_{}_(0,2).png'.format(i, test_name))
      plot_eigenvectors(raw_data[:,:2],
                        vecs,
                        labels=[int(j) for j in manifolds[i]],
                        filename='./images/manifold{}_{}_(0,1).png'.format(i, test_name))
      plot_eigenvectors(raw_data[:,1:],
                        vecs,
                        labels=[int(j) for j in manifolds[i]],
                        filename='./images/manifold{}_{}_(1,2).png'.format(i, test_name))

if __name__ == '__main__':
  main()
