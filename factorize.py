# use to run the algorithm on various datasets and generate figures
# for experiments.

import time
import os
import sys
import json
import pickle
import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import *
from plots import *
from algorithm import factorize

def preprocess_cryo_em_data(image_data, random_state=0):
  n_samples = image_data.shape[0]

  # apply PCA and standard scaling to the data
  print("\nApplying PCA and standard scaling...")
  t0 = time.perf_counter()
  image_data_ = np.reshape(image_data, (n_samples, -1))
  pca = PCA(n_components=4, random_state=random_state)
  image_data_ = pca.fit_transform(image_data_)

  # standard scale each channel
  scaler = StandardScaler()
  image_data_ = np.reshape(image_data_, (-1, image_data_.shape[1]))
  image_data_ = scaler.fit_transform(image_data_)
  image_data_ = np.reshape(image_data_, (n_samples, -1))
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  return image_data_

def do_generate_plots(image_dir, info, result, eig_crit, sim_crit):
  name = info['name']
  datatype = info['datatype']
  phi = result['phi']
  Sigma = result['Sigma']
  data = result['data']
  C = result['C_matrix']
  all_sims = result['all_sims']

  n_eigenvectors = Sigma.shape[0]

  if datatype == 'cryo-em':
    dimensions = [2 * info['x_stretch'], 2 * info['y_stretch'], 90]
    if name == 'cryo-em_x-theta_noisy':
      dimensions = [dimensions[0], dimensions[2], 0] # y = 0
  elif datatype == 'torus':
    dimensions = [2 * np.pi, 2 * np.pi, 0]
  else:
    dimensions = info['dimensions']

  if datatype == 'torus':
    data_dimensions = [2 * info['dimensions'][0],
                       2 * info['dimensions'][0],
                       2 * info['dimensions'][1]]
  else:
    data_dimensions = dimensions

  # get ground truth data to make plots clearer
  if datatype == 'cryo-em':
    image_data = info['image_data']
    data_gt = info['raw_data']
    if name == 'cryo-em_x-theta_noisy':
      data_gt = data_gt[:,[0,2]]
  else:
    data_gt = get_gt_data(data, datatype)

  print("\nSaving plots to", image_dir, flush=True)

  # plot original data
  print("Plotting original data...")
  if datatype == 'cryo-em':
    plot_cryo_em_data(image_data[:4],
                      filename='{}/{}'.format(image_dir, '{}_original_data.png'.format(name)))
  else:
    plot_synthetic_data(data, data_dimensions, azim=-30, elev=30,
                        filename='{}/{}'.format(image_dir, '{}_original_data.png'.format(name)))

  # plot eigenvectors
  vecs = [phi[:,i] for i in range(n_eigenvectors)]
  num_eigs_to_plot = 25
  print("Plotting first {} eigenvectors...".format(num_eigs_to_plot))
  eigenvectors_filename = '{}/{}_eigenvalues_{}.png'.format(image_dir, name, str(num_eigs_to_plot))
  plot_eigenvectors(data_gt, dimensions, vecs[:num_eigs_to_plot],
                    labels=[int(i) for i in range(num_eigs_to_plot)],
                    title='Laplace Eigenvectors',
                    filename=eigenvectors_filename)

  # plot manifolds
  manifolds = result['manifolds']
  independent_vecs = []
  for manifold in manifolds:
    vecs = [phi[:,int(i)] for i in manifold]
    independent_vecs.append(vecs)

  if datatype == 'rectangle3d':
    elev, azim = [30, -30]
  else:
    elev, azim = [30, 60]

  for m,vecs in enumerate(independent_vecs):
    print("Plotting eigenvectors for manifold {}...".format(m + 1))
    plot_eigenvectors(data_gt, dimensions, vecs[:5],
                      full=False,
                      labels=[int(j) for j in manifolds[m]],
                      filename='{}/{}'.format(image_dir, 'manifold{}_{}.png'.format(m, name)),
                      offset_scale=0,
                      elev=elev,
                      azim=azim)

  # plot 2d and 3d laplacian eigenmaps of each manifold
  for m in range(len(manifolds)):
    print("Plotting 2d laplacian eigenmap embedding for manifold {}...".format(m + 1))
    plot_eigenmap(data_gt[:,m], phi, manifolds[m][:min(2, len(manifolds[m]))],
                   filename='{}/{}'.format(image_dir, 'eigenmap{}_{}_2d.png'.format(m, name)))
    print("Plotting 3d laplacian eigenmap embedding for manifold {}...".format(m + 1))
    plot_eigenmap(data_gt[:,m], phi, manifolds[m][:min(3, len(manifolds[m]))],
                   filename='{}/{}'.format(image_dir, 'eigenmap{}_{}_3d.png'.format(m, name)))

  # plot sim-scores of best triplets
  print("Plotting all triplet similarity scores...")
  plot_triplet_sims(all_sims, thresh=sim_crit,
                    filename='{}/{}'.format(image_dir, 'triplet_sim-scores_{}.png'.format(name)))

  # plot mixture eigenvector sim-scores
  mixtures = get_product_eigs(manifolds, n_eigenvectors)
  steps = [5, 95, 15]
  print("Plotting select product eigenvector similarity scores...")
  plot_product_sims(mixtures, phi, Sigma, steps,
                    filename='{}/{}'.format(image_dir, 'product-eig_sim-scores_{}.png'.format(name)))

  # plot C matrix organized by manifold
  print("Plotting separability matrix...")
  plot_C_matrix(manifolds, C=C,
                filename='{}/{}'.format(image_dir, '{}_sep_matrix.png'.format(name)))

def main():
  parser = argparse.ArgumentParser(description='Run experiments on geometric data.')
  parser.add_argument('--data', type=str, default=None, required=True, 
                      help='Path to pickle file containing information about the data.')
  parser.add_argument('--configs', type=str, default=None, required=True, 
                      help='Path to json file containing algorithm params.')
  parser.add_argument('--outdir', type=str, default=None, required=True, 
                      help='The directory in which to save the results.')
  parser.add_argument('--generate_plots', action='store_true', default=False,
                      help='Set to generate and save figures',
                      dest='generate_plots')
  arg = vars(parser.parse_args())

  # retrieve data
  info = pickle.load(open(arg['data'], "rb"))
  datatype = info['datatype']

  # load and unpack parameters
  with open(arg['configs']) as f:
    configs = json.load(f)

  print("\nParameters...")
  for item, amount in configs.items():
    print("{:15}:  {}".format(item, amount))

  sigma = configs['sigma']
  n_factors = configs['n_factors']
  n_eigenvectors = configs['n_eigenvectors']
  eig_crit = configs['eig_crit']
  sim_crit = configs['sim_crit']
  K = configs['K']
  seed = configs['seed']

  if datatype == 'cryo-em': # preprocess cryo-EM data
    image_data = info['image_data']
    image_data_ = preprocess_cryo_em_data(image_data)
    info['data'] = image_data_

  result = factorize(info['data'], sigma, n_eigenvectors, n_factors, eig_crit, sim_crit, 
                      K=K, seed=seed, verbose=True)

  # create output directory
  if not os.path.exists(arg['outdir']):
    os.makedirs(arg['outdir'])
  result_file = '{}/{}'.format(arg['outdir'], 'results.pkl')
  print('\nSaving results to file', result_file, flush=True)

  # save info dictionary using pickle
  with open(result_file, 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

  # generate plots
  if arg['generate_plots']:
    do_generate_plots(arg['outdir'], info, result, eig_crit, sim_crit)

if __name__ == "__main__":
  main()
