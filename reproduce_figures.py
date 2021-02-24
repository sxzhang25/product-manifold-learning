# Use this file to reproduce figures from the paper

import time
import pickle
import os
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

matplotlib.use('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from plots import *
from utils import *
from factorize import *

n_eigenvectors = 100
rect3d_dimensions = [1 + np.sqrt(np.pi), 1.5, 0.05]
cryo_em_x = 20
cube_dimensions = [1.0 + np.sqrt(np.pi), 1.5, 7.0]

def main():
  data_dir = './data'
  results_dir = './results'
  image_dir = './figures'
  if not os.path.exists(image_dir):
    os.makedirs(image_dir)
  print('\nSaving images in directory', image_dir, flush=True)

  rect3d_info = pickle.load(open('{}/rectangle3d_info.pkl'.format(data_dir), "rb"))
  torus_info = pickle.load(open('{}/torus_info.pkl'.format(data_dir), "rb"))
  cryo_em_info = pickle.load(open('{}/cryo-em_x-theta_noisy_info.pkl'.format(data_dir), "rb"))
  cube_info = pickle.load(open('{}/cube_info.pkl'.format(data_dir), "rb"))

  rect3d_results = pickle.load(open('{}/rectangle3d/results.pkl'.format(results_dir), "rb"))
  torus_results = pickle.load(open('{}/torus/results.pkl'.format(results_dir), "rb"))
  cryo_em_results = pickle.load(open('{}/cryo-em_x-theta_noisy/results.pkl'.format(results_dir), "rb"))
  cube_results = pickle.load(open('{}/cube/results.pkl'.format(results_dir), "rb"))

  rect3d_data = rect3d_info['data']
  rect3d_data_gt = get_gt_data(rect3d_data, 'rectangle3d')

  torus_data = torus_info['data']
  torus_data_gt = get_gt_data(torus_data, 'torus')

  cryo_em_image_data = cryo_em_info['image_data']
  cryo_em_data = preprocess_cryo_em_data(cryo_em_image_data)
  cryo_em_raw_data = cryo_em_info['raw_data'][:,[0,2]] # y = 0

  cube_data = cube_info['data']

  # figure 1
  print('\nGenerating Figure 1...', flush=True)
  x = np.linspace(0, 1.5 + np.sqrt(np.pi), 100)
  y = np.linspace(0, 1.0, 100)
  xv, yv = np.meshgrid(x, y)
  data = np.zeros((10000,2))
  count = 0
  for i in x:
      for j in y:
          data[count] = [i,j]
          count += 1
  
  W = calc_W(data, sigma=0.1)
  phi, Sigma = calc_vars(data, W, sigma=0.1, n_eigenvectors=15, uniform=True)

  fig, axs = plt.subplots(3, 5, figsize=(10,3))
  for r in range(3):
      for c in range(5):
          axs[r,c].axis('off')
          axs[r,c].set_aspect('equal', 'datalim')
          axs[r,c].scatter(data[:,0], data[:,1], c=phi[:,5*r+c], s=10, marker='s')
  plt.savefig('{}/rectangle_eigenvectors.png'.format(image_dir))
  plt.close()

  # figure 2
  print('Generating Figure 2...', flush=True)
  plot_synthetic_data(rect3d_data, rect3d_dimensions, azim=-30, elev=30,
                      filename='{}/rectangle3d_original_data.png'.format(image_dir))

  rect3d_manifolds = rect3d_results['manifolds']
  rect3d_independent_vecs = []
  rect3d_phi = rect3d_results['phi']
  for manifold in rect3d_manifolds:
    vecs = [rect3d_phi[:,int(i)] for i in manifold]
    rect3d_independent_vecs.append(vecs)

  for m,vecs in enumerate(rect3d_independent_vecs):
    plot_eigenvectors(rect3d_data_gt, rect3d_dimensions, vecs[:5],
                      full=False,
                      labels=[int(j) for j in rect3d_manifolds[m]],
                      filename='{}/manifold{}_rectangle3d.png'.format(image_dir, m),
                      offset_scale=0,
                      elev=30,
                      azim=-30)

  # figure 3
  print('Generating Figure 3...', flush=True)
  image_data = cryo_em_info['image_data']
  plot_cryo_em_data(image_data[:4],
                    filename='{}/cryo-em_x-theta_noisy_original_data.png'.format(image_dir))

  cryo_em_manifolds = cryo_em_results['manifolds']
  cryo_em_independent_vecs = []
  cryo_em_phi = cryo_em_results['phi']
  for manifold in cryo_em_manifolds:
    vecs = [cryo_em_phi[:,int(i)] for i in manifold]
    cryo_em_independent_vecs.append(vecs)

  for m,vecs in enumerate(cryo_em_independent_vecs):
    plot_eigenvectors(cryo_em_raw_data,
                      [2 * cryo_em_x, 90, 0],
                      vecs[:5],
                      full=False,
                      labels=[int(j) for j in cryo_em_manifolds[m]],
                      filename='{}/cryo-em_x-theta_noisy.png'.format(image_dir))

  # figure 4
  print('Generating Figure 4...', flush=True)
  rect3d_mixtures = get_product_eigs(rect3d_results['manifolds'], n_eigenvectors)
  steps = [5, 95, 15]
  plot_product_sims(rect3d_mixtures, rect3d_results['phi'], rect3d_results['Sigma'], steps,
                    filename='{}/product-eig_sim-scores_rectangle3d.png'.format(image_dir))

  # figure 6
  print('Generating Figure 6...', flush=True)
  torus_manifolds = torus_results['manifolds']
  torus_independent_vecs = []
  torus_phi = torus_results['phi']

  # manifold factorization (ours)
  for manifold in torus_manifolds:
    vecs = [torus_phi[:,int(i)] for i in manifold]
    torus_independent_vecs.append(vecs)

  for m in range(len(rect3d_manifolds)):
    plot_eigenmap(rect3d_data_gt[:,m], rect3d_phi, rect3d_manifolds[m][:2],
                   filename='{}/eigenmap{}_rectangle3d_2d.png'.format(image_dir, m))
  for m in range(len(torus_manifolds)):
    plot_eigenmap(torus_data_gt[:,m], torus_phi, torus_manifolds[m][:2],
                   filename='{}/eigenmap{}_torus_2d.png'.format(image_dir, m))
  for m in range(len(cryo_em_manifolds)):
    plot_eigenmap(cryo_em_raw_data[:,m], cryo_em_phi, cryo_em_manifolds[m][:2],
                   filename='{}/eigenmap{}_cryo-em_x-theta_noisy_2d.png'.format(image_dir, m))

  # diffusion maps
  plot_eigenmap(rect3d_data_gt[:,0], rect3d_phi, [1, 2, 3],
                filename='{}/diffusionmap{}_rectangle3d_2d.png'.format(image_dir, m))
  plot_eigenmap(torus_data_gt[:,0], torus_phi, [1, 2, 3],
                filename='{}/diffusionmap{}_torus_2d.png'.format(image_dir, m))
  plot_eigenmap(cryo_em_raw_data[:,0], cryo_em_phi, [1, 2, 3],
                filename='{}/diffusionmap{}_cryo-em_x-theta_noisy_2d.png'.format(image_dir, m))

  # linear ICA
  transformer = FastICA(n_components=3, random_state=0)
  rect3d_fastICA = transformer.fit_transform(rect3d_data)
  plot_fast_ica(rect3d_fastICA, rect3d_data_gt[:,0], filename='{}/fastICA_rect3d.png'.format(image_dir, m))
  torus_fastICA = transformer.fit_transform(torus_data)
  plot_fast_ica(torus_fastICA, torus_data_gt[:,0], filename='{}/fastICA_torus.png'.format(image_dir, m))
  cryo_em_fastICA = transformer.fit_transform(cryo_em_data)
  plot_fast_ica(cryo_em_fastICA, cryo_em_raw_data[:,0], filename='{}/fastICA_cryo_em.png'.format(image_dir, m))

  # figure 7
  print('Generating Figure 7...', flush=True)
  with open('configs/configs_cube.json') as f:
      configs_cube = json.load(f)

  # first factorization
  cube_manifolds = cube_results['manifolds']
  cube_phi = cube_results['phi']
  cube_independent_vecs = []
  for manifold in cube_manifolds:
    vecs = [cube_phi[:,int(i)] for i in manifold]
    cube_independent_vecs.append(vecs)

  for m,vecs in enumerate(cube_independent_vecs):
    plot_eigenvectors(cube_data, cube_dimensions, vecs[:5],
                      full=False,
                      labels=[int(j) for j in cube_manifolds[m]],
                      filename='{}/{}'.format(image_dir, 'manifold{}_cube_round1.png'.format(m)),
                      offset_scale=0)

  # second factorization
  cube_results_two = factorize(cube_info['data'], configs_cube['sigma'], 
                      configs_cube['n_eigenvectors'], configs_cube['n_factors'], 
                      configs_cube['eig_crit'], configs_cube['sim_crit'],
                      uniform=True, K=configs_cube['K'], seed=configs_cube['seed'], exclude_eigs=cube_manifolds[1], verbose=False)

  cube_manifolds = cube_results_two['manifolds']
  cube_phi = cube_results_two['phi']
  cube_independent_vecs = []
  for manifold in cube_manifolds:
      vecs = [cube_phi[:,int(i)] for i in manifold]
      cube_independent_vecs.append(vecs)

  for m,vecs in enumerate(cube_independent_vecs):
      plot_eigenvectors(cube_data, cube_dimensions, vecs[:5],
                        full=False,
                        labels=[int(j) for j in cube_manifolds[m]],
                        filename='{}/{}'.format(image_dir, 'manifold{}_cube_round2.png'.format(m)),
                        offset_scale=0)
  


if __name__ == "__main__":
  main()
