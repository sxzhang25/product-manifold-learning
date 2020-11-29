# Use this file to reproduce figures from the paper

import time
import pickle
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plots import *
from utils import *

n_eigenvectors = 100
rect3d_dimensions = [1 + np.sqrt(np.pi), 1.5, 0.05]
cryo_em_x = 20

def main():
  data_dir = './data/'
  image_dir = './images/'
  if not os.path.exists(image_dir):
    os.makedirs(image_dir)

  rect3d_info = pickle.load(open(data_dir + 'rectangle3d_info.pickle', "rb"))
  torus_info = pickle.load(open(data_dir + 'torus_info.pickle', "rb"))
  cryo_em_info = pickle.load(open(data_dir + 'cryo-em_x-theta_noisy_info.pickle', "rb"))

  rect3d_data = rect3d_info['data']
  rect3d_data_gt = get_gt_data(rect3d_data, 'rectangle3d')

  torus_data = torus_info['data']
  torus_data_gt = get_gt_data(torus_data, 'torus')

  cryo_em_image_data = cryo_em_info['image_data']
  cryo_em_raw_data = cryo_em_info['raw_data'][:,[0,2]] # y = 0

  # figure 1
  plot_synthetic_data(rect3d_data, rect3d_dimensions, azim=-30, elev=30,
                      filename=image_dir + 'rectangle3d_original_data.png')

  rect3d_manifolds = rect3d_info['manifolds']
  rect3d_independent_vecs = []
  rect3d_phi = rect3d_info['phi']
  for manifold in rect3d_manifolds:
    vecs = [rect3d_phi[:,int(i)] for i in manifold]
    rect3d_independent_vecs.append(vecs)

  for m,vecs in enumerate(rect3d_independent_vecs):
    print("Plotting eigenvectors for manifold {}...".format(m + 1))
    plot_eigenvectors(rect3d_data_gt, rect3d_dimensions, vecs[:5],
                      full=False,
                      labels=[int(j) for j in rect3d_manifolds[m]],
                      filename=image_dir + 'manifold{}_rectangle3d.png'.format(m),
                      offset_scale=0,
                      elev=30,
                      azim=-30)

  # figure 2
  image_data = cryo_em_info['image_data']
  plot_cryo_em_data(image_data[:4],
                    filename=image_dir + 'cryo-em_x-theta_noisy_original_data.png')

  cryo_em_manifolds = cryo_em_info['manifolds']
  cryo_em_independent_vecs = []
  cryo_em_phi = cryo_em_info['phi']
  for manifold in cryo_em_manifolds:
    vecs = [cryo_em_phi[:,int(i)] for i in manifold]
    cryo_em_independent_vecs.append(vecs)

  for m,vecs in enumerate(cryo_em_independent_vecs):
    plot_eigenvectors(cryo_em_raw_data,
                      [2 * cryo_em_x, 90, 0],
                      vecs[:5],
                      full=False,
                      labels=[int(j) for j in cryo_em_manifolds[m]],
                      filename=image_dir + 'manifold{}_cryo-em_x-theta_noisy.png'.format(m))

  # figure 3
  rect3d_mixtures = get_product_eigs(rect3d_info['manifolds'], n_eigenvectors)
  steps = [5, 95, 15]
  plot_product_sims(rect3d_mixtures, rect3d_info['phi'], rect3d_info['Sigma'], steps,
                    filename=image_dir + 'product-eig_sim-scores_rectangle3d.png')

  # figure 4
  torus_manifolds = torus_info['manifolds']
  torus_independent_vecs = []
  torus_phi = torus_info['phi']
  for manifold in torus_manifolds:
    vecs = [torus_phi[:,int(i)] for i in manifold]
    torus_independent_vecs.append(vecs)

  for m in range(len(rect3d_manifolds)):
    plot_eigenmap(rect3d_data_gt[:,m], rect3d_phi, rect3d_manifolds[m][:2],
                   filename=image_dir + 'eigenmap{}_rectangle3d_2d.png'.format(m))
  for m in range(len(torus_manifolds)):
    plot_eigenmap(torus_data_gt[:,m], torus_phi, torus_manifolds[m][:2],
                   filename=image_dir + 'eigenmap{}_torus_2d.png'.format(m))
  for m in range(len(cryo_em_manifolds)):
    plot_eigenmap(cryo_em_raw_data[:,m], cryo_em_phi, cryo_em_manifolds[m][:2],
                   filename=image_dir + 'eigenmap{}_cryo-em_x-theta_noisy_2d.png'.format(m))

if __name__ == "__main__":
  main()
