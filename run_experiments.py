# use to run the algorithm on various datasets and generate figures
# for experiments.

import time
import os
import sys
import json
import pickle
import argparse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import *
from plots import *
from algorithm import *

def preprocess_cryo_em_data(image_data):
  n_samples = image_data.shape[0]

  # apply PCA and standard scaling to the data
  print("\nApplying PCA and standard scaling...")
  t0 = time.perf_counter()
  image_data_ = np.reshape(image_data, (n_samples, -1))
  pca = PCA(n_components=4)
  image_data_ = pca.fit_transform(image_data_)

  # standard scale each channel
  scaler = StandardScaler()
  image_data_ = np.reshape(image_data_, (-1, image_data_.shape[1]))
  image_data_ = scaler.fit_transform(image_data_)
  image_data_ = np.reshape(image_data_, (n_samples, -1))
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  return image_data_

def do_generate_plots(image_dir, info, eig_crit, sim_crit):
  name = info['name']
  datatype = info['datatype']
  phi = info['phi']
  Sigma = info['Sigma']
  data = info['data']
  C = info['C_matrix']
  all_sims = info['all_sims']

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

  print("\nGenerating plots...")

  # plot original data
  print("Plotting original data...")
  if datatype == 'cryo-em':
    plot_cryo_em_data(image_data[:4],
                      filename=image_dir + '{}_original_data.pdf'.format(name))
  else:
    plot_synthetic_data(data, data_dimensions, azim=-30, elev=30,
                        filename=image_dir + '{}_original_data.pdf'.format(name))

  # plot eigenvectors
  vecs = [phi[:,i] for i in range(n_eigenvectors)]
  num_eigs_to_plot = 25
  print("Plotting first {} eigenvectors...".format(num_eigs_to_plot))
  eigenvectors_filename = image_dir + name + '_eigenvalues_' + str(num_eigs_to_plot) + '.pdf'
  plot_eigenvectors(data_gt, dimensions, vecs[:num_eigs_to_plot],
                    labels=[int(i) for i in range(num_eigs_to_plot)],
                    title='Laplace Eigenvectors',
                    filename=eigenvectors_filename)

  # plot manifolds
  manifolds = info['manifolds']
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
                      filename=image_dir + 'manifold{}_{}.pdf'.format(m, name),
                      offset_scale=0,
                      elev=elev,
                      azim=azim)

  # plot 2d and 3d laplacian eigenmaps of each manifold
  for m in range(len(manifolds)):
    print("Plotting 2d laplacian eigenmap embedding for manifold {}...".format(m + 1))
    plot_eigenmap(data_gt[:,m], phi, manifolds[m][:min(2, len(manifolds[m]))],
                   filename=image_dir + 'eigenmap{}_{}_2d.pdf'.format(m, name))
    print("Plotting 3d laplacian eigenmap embedding for manifold {}...".format(m + 1))
    plot_eigenmap(data_gt[:,m], phi, manifolds[m][:min(3, len(manifolds[m]))],
                   filename=image_dir + 'eigenmap{}_{}_3d.pdf'.format(m, name))

  # plot sim-scores of best triplets
  print("Plotting all triplet similarity scores...")
  plot_triplet_sims(all_sims, thresh=sim_crit,
                    filename=image_dir + 'triplet_sim-scores_{}.pdf'.format(name))

  # plot mixture eigenvector sim-scores
  mixtures = get_product_eigs(manifolds, n_eigenvectors)
  steps = [5, 95, 15]
  print("Plotting select product eigenvector similarity scores...")
  plot_product_sims(mixtures, phi, Sigma, steps,
                    filename=image_dir + 'product-eig_sim-scores_{}.pdf'.format(name))

  # plot C matrix organized by manifold
  print("Plotting separability matrix...")
  plot_C_matrix(manifolds, C=C,
                filename=image_dir + '{}_sep_matrix.pdf'.format(name))

def main():
  parser = argparse.ArgumentParser(description='Run experiments on geometric data.')
  parser.add_argument("file_list", nargs=2)
  parser.add_argument('--generate_plots', action='store_true', default=False,
                      help='Set to True to generate and save figures',
                      dest='generate_plots')
  generate_plots = parser.parse_args().generate_plots
  data_file = parser.parse_args().file_list[0]
  configs_file = parser.parse_args().file_list[1]

  # retrieve data
  info = pickle.load(open(data_file, "rb"))
  datatype = info['datatype']

  # load and unpack parameters
  with open(configs_file) as f:
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

  if datatype == 'cryo-em':
    info = run_algorithm(info, sigma, n_eigenvectors, n_factors, eig_crit, sim_crit,
                         uniform=False, K=K, seed=seed, verbose=True)
  else:
    info = run_algorithm(info, sigma, n_eigenvectors, n_factors, eig_crit, sim_crit,
                         uniform=True, K=K, seed=seed, verbose=True)

  # save info dictionary using pickle
  print("\nSaving data...")
  with open(data_file, 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

  # generate plots
  if generate_plots:
    image_dir = './images/'
    if not os.path.exists(image_dir):
      os.makedirs(image_dir)
    do_generate_plots(image_dir, info, eig_crit, sim_crit)

if __name__ == "__main__":
  main()
