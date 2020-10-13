import sys
import json
import pickle
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cryo_em import *
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
  K = params['K']

  generate_plots = True # if set to true, plots will be created and saved

  # generate or load data
  data_dir = './data/' # make sure this directory exists!
  data_filename = data_dir + '{}_info.pickle'.format(test_name)
  if precomputed:
    # data has already been generated and is stored in pickle file
    info = pickle.load(open(data_filename, "rb"))

    # load data
    print("\nLoading data...")
    image_data = info['image_data']
    raw_data = info['raw_data']

  else:
    # create a dictionary to store all information in
    info = {}

    # generate random data
    print("\nGenerating random data...")
    image_data, raw_data = generate_cryo_em_data(n_samples, x=x, y=y, var=var)
    info['image_data'] = image_data
    info['raw_data'] = raw_data

  if test_name == "cryo_em_x,theta" or test_name == "cryo_em_x,theta_noisy":
    raw_data = raw_data[:,[0,2]]

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
  phi, Sigma = syn.calc_vars(image_data_, W, sigma, n_eigenvectors=n_eigenvectors)
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

  if generate_plots:
    print("\nGenerating plots...")

    # plot original data
    print("Plotting original data...")
    plot_cryo_em_data(image_data[:4],
              filename='./images/{}_original_data.png'.format(test_name))

    # plot eigenvectors
    vecs = [phi[:,i] for i in range(n_eigenvectors)]
    num_eigs_to_plot = 25
    print("Plotting first {} eigenvectors...".format(num_eigs_to_plot))
    eigenvectors_filename = './images/' + test_name + '_eigenvalues_' + str(num_eigs_to_plot) + '.png'
    plot_eigenvectors(raw_data, [x, 90, 0], vecs[:num_eigs_to_plot],
                      labels=[int(i) for i in range(num_eigs_to_plot)],
                      title='Laplace Eigenvectors',
                      filename=eigenvectors_filename )

    # plot manifolds
    manifolds = info['manifolds']
    independent_vecs = []
    for manifold in manifolds:
      vecs = [phi[:,int(i)] for i in manifold]
      independent_vecs.append(vecs)

    for m,vecs in enumerate(independent_vecs):
      print("Plotting eigenvectors for manifold {}...".format(m + 1))
      plot_eigenvectors(raw_data,
                        [2 * x, 90, 0],
                        vecs[:5],
                        full=False,
                        labels=[int(j) for j in manifolds[m]],
                        filename='./images/manifold{}_{}.png'.format(m, test_name))

    # plot 2d and 3d laplacian eigenmaps of each manifold
    for m in range(len(manifolds)):
      print("Plotting 2d laplacian eigenmap for manifold {}...".format(m + 1))
      plot_embedding(phi, manifolds[m][:min(2, len(manifolds[0]))],
                     filename='./images/embedding{}_{}_2d.png'.format(m, test_name))
      print("Plotting 3d laplacian eigenmap for manifold {}...".format(m + 1))
      plot_embedding(phi, manifolds[m][:min(3, len(manifolds[0]))],
                     filename='./images/embedding{}_{}_3d.png'.format(m, test_name))

    # plot correlations of best triplets
    print("Plotting all triplet correlations...")
    plot_triplet_correlations(all_corrs, thresh=corr_thresh,
                              filename='./images/triplet_correlations_{}.png'.format(test_name))

    # plot mixture eigenvector correlations
    mixtures = get_mixture_eigenvectors(manifolds, n_eigenvectors)
    steps = [5, 95, 18]
    print("Plotting select mixture correlations...")
    plot_mixture_correlations(mixtures, phi, Sigma, steps,
                              filename='./images/mixture_correlations_{}.png'.format(test_name))

if __name__ == '__main__':
  main()
