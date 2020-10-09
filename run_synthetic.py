import time
import sys
import json
import pickle

from synthetic import *
from plots import *

def main():
  # load and unpack parameters
  datafile = sys.argv[1]
  with open(datafile) as f:
    params = json.load(f)

  print("\nParameters...")
  for item, amount in params.items():
    print("{:15}:  {}".format(item, amount))

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
  K = params['K']

  generate_plots = True # if set to true, plots will be created and saved

  # generate or load data
  data_dir = './data/'
  data_filename = data_dir + '{}_info.pickle'.format(test_name)
  if precomputed:
    # data has already been generated and is stored in pickle file
    info = pickle.load(open(data_filename, "rb"))

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

    # get the 'ground truth' data (true manifolds used to generate data)
    data_gt = get_gt_data(data, datatype)

  else:
    # create a dictionary to store all information in
    info = {}

    # generate random data
    print("\nGenerating random data...")
    data = generate_synthetic_data(dimensions,
                                   noise=noise,
                                   n_samples=n_samples,
                                   datatype=datatype,
                                   seed=seed)
    info['data'] = data
    data_gt = get_gt_data(data, datatype)

    # compute eigenvectors
    print("\nComputing eigenvectors...")
    t0 = time.perf_counter()
    W = calc_W(data, sigma)
    phi, Sigma = calc_vars(data, W, sigma, n_eigenvectors=n_eigenvectors)
    t1 = time.perf_counter()
    print("  Time: %2.2f seconds" % (t1-t0))

    info['phi'] = phi
    info['Sigma'] = Sigma

  np.random.seed(255)

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
  labels = split_eigenvectors(best_matches, best_corrs, n_eigenvectors, K, n_comps=n_comps, verbose=True)
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
  with open(data_filename, 'wb') as handle:
    pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print("Done")

  # generate plots
  if generate_plots:
    print("\nGenerating plots...")
    image_dir = './images/'

    # plot original data
    print("Plotting original data...")
    if datatype == "torus":
      dimensions = [2 * dimensions[0], 2 * dimensions[0], 2 * dimensions[1]]
    plot_synthetic_data(data, dimensions,
                        filename=image_dir + '{}_original_data.png'.format(test_name))

    # plot eigenvectors
    vecs = [phi[:,i] for i in range(n_eigenvectors)]
    num_eigs_to_plot = 25
    print("Plotting first {} eigenvectors...".format(num_eigs_to_plot))
    eigenvectors_filename = image_dir + test_name + '_eigenvalues_' + str(num_eigs_to_plot) + '.png'
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

    for m,vecs in enumerate(independent_vecs):
      print("Plotting eigenvectors for manifold {}...".format(m + 1))
      plot_eigenvectors(data_gt, dimensions, vecs[:5],
                        full=False,
                        labels=[int(j) for j in manifolds[m]],
                        filename=image_dir + 'manifold{}_{}.png'.format(m, test_name),
                        offset_scale=0,
                        elev=30,
                        azim=-30)

    # plot 2d and 3d laplacian eigenmaps of each manifold
    for m in range(len(manifolds)):
      print("Plotting 2d laplacian eigenmap for manifold {}...".format(m + 1))
      plot_embedding(phi, manifolds[m][:min(2, len(manifolds[0]))],
                     filename=image_dir + 'embedding{}_{}_2d.png'.format(m, test_name))
      print("Plotting 3d laplacian eigenmap for manifold {}...".format(m + 1))
      plot_embedding(phi, manifolds[m][:min(3, len(manifolds[0]))],
                     filename=image_dir + 'embedding{}_{}_3d.png'.format(m, test_name))

    # plot correlations of best triplets
    print("Plotting all triplet correlations...")
    plot_triplet_correlations(all_corrs, thresh=corr_thresh,
                              filename=image_dir + 'triplet_correlations_{}.png'.format(test_name))

    # plot mixture eigenvector correlations
    mixtures = get_mixture_eigenvectors(manifolds, n_eigenvectors)
    steps = [5, 95, 18]
    print("Plotting select mixture correlations...")
    plot_mixture_correlations(mixtures, phi, Sigma, steps,
                              filename=image_dir + 'mixture_correlations_{}.png'.format(test_name))

if __name__ == "__main__":
  main()
