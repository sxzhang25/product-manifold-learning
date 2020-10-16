# Use this file to reproduce figures from the paper
import argparse
import time
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plots import *
from synthetic import *
from cryo_em import *

n_comps = 2
n_eigenvectors = 100
n_samples = 10_000

rect3d_dimensions = [1 + np.sqrt(np.pi), 1.5, 0.05]
torus_dimensions = [1 + np.sqrt(np.pi), 2]
cryo_em_x = 20
cryo_em_y = 0
cryo_em_var = 10000

def generate_rect3d_data(data_dir):
  sigma = 0.5
  
  info = {}
  print("\nGenerating rectangle3d data...")
  data = generate_synthetic_data(rect3d_dimensions,
                                 noise=0,
                                 n_samples=n_samples,
                                 datatype='rectangle3d',
                                 seed=0)
  info['data'] = data

  print("\nComputing eigenvectors...")
  t0 = time.perf_counter()
  W = calc_W(data, sigma)
  phi, Sigma = calc_vars(data, W, sigma, n_eigenvectors=100)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  info['phi'] = phi
  info['Sigma'] = Sigma
  
  print("\nSaving data...")
  data_filename = data_dir + 'rectangle3d_info.pickle'
  with open(data_filename, 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")
  
def generate_torus_data(data_dir):
  sigma = 0.5
  
  info = {}
  print("\nGenerating torus data...")
  data = generate_synthetic_data(torus_dimensions,
                                 noise=0,
                                 n_samples=n_samples,
                                 datatype='torus',
                                 seed=0)
  info['data'] = data
  
  print("\nComputing eigenvectors...")
  t0 = time.perf_counter()
  W = calc_W(data, sigma)
  phi, Sigma = calc_vars(data, W, sigma, n_eigenvectors=100)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  info['phi'] = phi
  info['Sigma'] = Sigma
  
  print("\nSaving data...")
  data_filename = data_dir + 'torus_info.pickle'
  with open(data_filename, 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

def run_synthetic_experiments(data_dir, info, test_name, lambda_thresh, corr_thresh):
  np.random.seed(255)
  
  phi, Sigma = info['phi'], info['Sigma']
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
  labels, C = split_eigenvectors(best_matches, best_corrs, n_eigenvectors,
                                 n_comps=n_comps, verbose=True)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  # save manifolds
  print("\nManifolds...")
  manifolds = []
  for m in range(n_comps):
    manifold = labels[0][np.where(labels[1]==m)[0]]
    manifolds.append(manifold)
    print("Manifold #{}".format(m + 1), manifold)

  info['manifolds'] = manifolds
  
  print("\nSaving data...")
  data_filename = data_dir + '{}_info.pickle'.format(test_name)
  with open(data_filename, 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")
  
  return info

def generate_noisy_cryo_em_data(data_dir):  
  info = {}
  print("\nGenerating cryo-EM data...")
  image_data, raw_data = generate_cryo_em_data(n_samples, 
                                               x=cryo_em_x, 
                                               y=cryo_em_y, 
                                               var=cryo_em_var)
  info['image_data'] = image_data
  info['raw_data'] = raw_data
  
  print("\nSaving data...")
  data_filename = data_dir + 'cryo-em_x-theta_noisy_info.pickle'
  with open(data_filename, 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")
  
def run_cryo_em_experiments(data_dir, info):
  sigma = 10
  
  image_data = info['image_data']
  raw_data = info['raw_data'][:,[0,2]]

  np.random.seed(255)
  
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

  # compute eigenvectors
  print("\nComputing eigenvectors...")
  t0 = time.perf_counter()
  W = syn.calc_W(image_data_, sigma)
  phi, Sigma = syn.calc_vars(image_data_, W, sigma, n_eigenvectors=n_eigenvectors, uniform=False)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  info['phi'] = phi
  info['Sigma'] = Sigma

  # find combos
  print("\nComputing combos...")
  t0 = time.perf_counter()
  best_matches, best_corrs, all_corrs = find_combos(phi, Sigma, n_comps, lambda_thresh=1, corr_thresh=0.8)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))
  info['best_matches'] = best_matches
  info['best_corrs'] = best_corrs
  info['all_corrs'] = all_corrs

  # split eigenvectors
  print("\nSplitting eigenvectors...")
  t0 = time.perf_counter()
  labels, C = split_eigenvectors(best_matches, best_corrs, n_eigenvectors,
                                 n_comps=n_comps, verbose=True)
  t1 = time.perf_counter()
  print("  Time: %2.2f seconds" % (t1-t0))

  # save manifolds
  print("\nManifolds...")
  manifolds = []
  for m in range(n_comps):
    manifold = labels[0][np.where(labels[1]==m)[0]]
    manifolds.append(manifold)
    print("Manifold #{}".format(m + 1), manifold)

  info['manifolds'] = manifolds

  # save info dictionary using pickle
  print("\nSaving data...")
  data_filename = data_dir + 'cryo-em_x-theta_noisy_info.pickle'
  with open(data_filename, 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")
  
  return info  

def main():
  parser = argparse.ArgumentParser(description='Run paper experiments.')
  parser.add_argument('--generate_data', action='store_true', default=False,
                      help='Set True if data has already been precomputed',
                      dest='generate_data')
  generate_data = parser.parse_args().generate_data
  
  data_dir = './test_data/'
  if generate_data:
    generate_rect3d_data(data_dir)
    generate_torus_data(data_dir)
    generate_noisy_cryo_em_data(data_dir)
  
  rect3d_info = pickle.load(open(data_dir + 'rectangle3d_info.pickle', "rb"))
  torus_info = pickle.load(open(data_dir + 'torus_info.pickle', "rb"))
  cryo_em_info = pickle.load(open(data_dir + 'cryo-em_x-theta_noisy_info.pickle', "rb"))
  
  print("\nRunning rectangle3D experiments")
  rect3d_info = run_synthetic_experiments(data_dir, rect3d_info, 'rectangle3d', 0.5, 0.85)
  rect3d_data = rect3d_info['data']
  rect3d_data_gt = get_gt_data(rect3d_data, 'rectangle3d')
  
  print("\nRunning torus experiments")
  torus_info = run_synthetic_experiments(data_dir, torus_info, 'torus', 1.0, 0.6)
  torus_data = torus_info['data']
  torus_data_gt = get_gt_data(torus_data, 'torus')
  
  print("\nRunning cryo-EM experiments")
  cryo_em_info = run_cryo_em_experiments(data_dir, cryo_em_info)
  cryo_em_image_data = cryo_em_info['image_data']
  cryo_em_raw_data = cryo_em_info['raw_data'][:,[0,2]] # y = 0
  
  image_dir = './images/'
  
  # figure 1
  plot_synthetic_data(rect3d_data, rect3d_dimensions, azim=-30, elev=30,
                      filename=image_dir + 'rectangle3d_original_data.pdf')
  
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
                      filename=image_dir + 'manifold{}_rectangle3d.pdf'.format(m),
                      offset_scale=0,
                      elev=30,
                      azim=-30)
  
  # figure 2
  image_data = cryo_em_info['image_data']
  plot_cryo_em_data(image_data[:4],
                    filename=image_dir + 'cryo-em_x-theta_noisy_original_data.pdf')
  
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
                      filename=image_dir + 'manifold{}_cryo-em_x-theta_noisy.pdf'.format(m))
  
  # figure 3
  rect3d_mixtures = get_mixture_eigenvectors(rect3d_info['manifolds'], n_eigenvectors)
  steps = [5, 95, 15]
  plot_mixture_correlations(rect3d_mixtures, rect3d_info['phi'], rect3d_info['Sigma'], steps,
                            filename=image_dir + 'mixture_correlations_rectangle3d.pdf')
  
  # figure 4
  torus_manifolds = torus_info['manifolds']
  torus_independent_vecs = []
  torus_phi = torus_info['phi']
  for manifold in torus_manifolds:
    vecs = [torus_phi[:,int(i)] for i in manifold]
    torus_independent_vecs.append(vecs)
  
  for m in range(len(rect3d_manifolds)):
    plot_embedding(rect3d_data_gt[:,m], rect3d_phi, rect3d_manifolds[m][:2],
                   filename=image_dir + 'embedding{}_rectangle3d_2d.pdf'.format(m))
  for m in range(len(torus_manifolds)):
    plot_embedding(torus_data_gt[:,m], torus_phi, torus_manifolds[m][:2],
                   filename=image_dir + 'embedding{}_torus_2d.pdf'.format(m))
  for m in range(len(cryo_em_manifolds)):
    plot_embedding(cryo_em_raw_data[:,m], cryo_em_phi, cryo_em_manifolds[m][:2],
                   filename=image_dir + 'embedding{}_cryo-em_x-theta_noisy_2d.pdf'.format(m))

if __name__ == "__main__":
  main()