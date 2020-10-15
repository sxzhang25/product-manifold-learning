import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
from scipy.linalg import block_diag

import synthetic as syn

###
# PLOT DATA
###

def plot_synthetic_data(data, dimensions, title=None, filename=None, azim=60, elev=30, proj_type='persp'):
  '''
  plot the original data
  '''

  fig = plt.figure(figsize=(5,5))
  if data.shape[1] <= 2:
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=5)
  elif data.shape[1] == 3:
    ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev, proj_type=proj_type)
    x, y, z = dimensions
    axis_max = np.max(dimensions)
    ax.set_xlim3d((x - axis_max) / 2, (x + axis_max) / 2)
    ax.set_ylim3d((y - axis_max) / 2, (y + axis_max) / 2)
    ax.set_zlim3d((z - axis_max) / 2, (z + axis_max) / 2)
    ax.scatter(data[:,0], data[:,1], data[:,2], s=5, c=data[:,2])
  else:
    ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev, proj_type=proj_type)
    x, y, z = dimensions
    axis_max = np.max(dimensions)
    ax.set_xlim3d((x - axis_max) / 2, (x + axis_max) / 2)
    ax.set_ylim3d((y - axis_max) / 2, (y + axis_max) / 2)
    ax.set_zlim3d((z - axis_max) / 2, (z + axis_max) / 2)
    g = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], s=5)
    cb = plt.colorbar(g)
  if title:
    plt.title(title, pad=10)
  if filename:
    plt.savefig(filename)
  plt.show()

def plot_cryo_em_data(data, title=None, filename=None):
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
        ax.imshow(vol, cmap='gray')
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
  if title:
    plt.title(title, pad=10)
  if filename:
    plt.savefig(filename)
  plt.show()

###
# PLOT EIGENVECTORS
###

def plot_eigenvector(data, v, scaled=False, title=None, filename=None):
  '''
  plots the eigenvector v
  if scale=True, then scale v to [0,1]
  if index is specified, label the graph
  '''

  vs = scale(v, [0,1]) if scaled else v
  fig = plt.figure()
  ax = fig.add_subplot(111)
  g = ax.scatter(data[:,0], data[:,1], marker="s", c=vs)
  ax.set_aspect('equal', 'datalim')
  cb = plt.colorbar(g)
  if title:
    plt.title(title)
  if filename:
    plt.savefig(filename)
  plt.show()

def plot_eigenvectors(data, dimensions, eigenvectors, full=True, labels=None, title=None,
                      filename=None, offset_scale=0.1, azim=60, elev=30, proj_type='persp'):
  '''
  plots the array eigenvectors

  eigenvectors: a list of eigenvectors
  '''
  rows = int(np.ceil(len(eigenvectors)**0.5))
  cols = rows
  if data.shape[1] <= 2:
    if full:
      fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
      for r in range(rows):
        for c in range(cols):
          ax = axs[r, c] if len(eigenvectors) > 1 else axs
          index = r * cols + c
          if index >= len(eigenvectors):
            ax.set_visible(False)
          else:
            v = eigenvectors[r * cols + c]
            v /= np.linalg.norm(v)
            g = ax.scatter(data[:,0], data[:,1], marker="s", c=v)
            fig.colorbar(g, ax=ax)
            ax.set_aspect('equal', 'datalim')
            if labels is not None:
              ax.set_title(labels[index])
    else:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev, proj_type=proj_type)
      x, y, z = dimensions
      axis_max = np.max(dimensions)
      offset = offset_scale * len(eigenvectors)
      ax.set_xlim3d((x - axis_max) / 2, (x + axis_max) / 2 + offset)
      ax.set_ylim3d((y - axis_max) / 2, (y + axis_max) / 2 + offset)
      ax.set_zlim3d(0, len(eigenvectors))
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_zticks([])
      plt.axis('off')
      plt.grid(b=None)
      plt.tight_layout()
      plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

      for i in range(len(eigenvectors)):
        v = eigenvectors[i]
        v /= np.linalg.norm(v)
        g = ax.scatter(data[:,0] + offset_scale * np.ones(data.shape[0]) * (len(eigenvectors) - i),
                       data[:,1] + offset_scale * np.ones(data.shape[0]) * (len(eigenvectors) - i),
                       np.ones(data.shape[0]) * (len(eigenvectors) - i),
                       marker="s",
                       c=v)
  else:
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    for r in range(rows):
      for c in range(cols):
        index = r * cols + c
        ax = fig.add_subplot(rows, cols, index+1, projection='3d')
        if index >= len(eigenvectors):
          ax.set_visible(False)
        else:
          v = eigenvectors[r * cols + c]
          v /= np.linalg.norm(v)
          g = ax.scatter(data[:,0], data[:,1], data[:,2], marker="s", c=v)
          fig.colorbar(g, ax=ax)
          if labels is not None:
            ax.set_title(labels[index])
  if title:
    plt.title(title)
  if filename:
    plt.savefig(filename)
  plt.show()

def plot_eigenvectors_combo(data, v1, v2, title=None, filename=None):
  '''
  plots the combination (element-wise multiplication) of vectors v1 and v2
  '''

  v = v1 * v2
  plot_eigenvector(data, v, title, filename)

def plot_independent_eigenvectors(manifold1, manifold2,
                                  n_eigenvectors,
                                  title=None, filename=None):
  '''
  plots a matrix with the independent eigenvectors color-coded by manifold
  '''

  # create separate manifolds
  data = np.zeros((1,n_eigenvectors))
  for i in manifold1:
    data[0,int(i)] = 1
  for j in manifold2:
    data[0,int(j)] = -1

  # create discrete colormap
  if 1 in manifold1:
    cmap = colors.ListedColormap(['blue', 'white', 'red'])
  else:
    cmap = colors.ListedColormap(['red', 'white', 'blue'])
  bounds = [-0.5,-0.25,0.25,0.5]
  norm = colors.BoundaryNorm(bounds, cmap.N)

  fig, ax = plt.subplots()
  ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

  # draw gridlines
  ax.axes.get_yaxis().set_visible(False)
  ax.set_xticks(np.arange(0, n_eigenvectors, 5));

  if title:
    plt.title(title, fontsize=20)
  if filename:
    plt.savefig(filename)
  plt.show()

def plot_triplet_correlations(corrs, thresh=None, filename=None):
  counts, bins, patches = plt.hist(corrs.values(), 50, density=True)
  if thresh:
    plt.plot([thresh, thresh], [0, np.max(counts)], "k--", linewidth=1)
    plt.annotate('thresh = {}'.format(thresh), xy=(thresh + 0.01, np.max(counts)))
  plt.yticks([])
  plt.title('Correlation density')
  if filename:
    plt.savefig(filename)
  plt.show()

def plot_embedding(data_gt, phi, dims, filename=None):
  if len(dims) is 0:
    return
  fig = plt.figure()
  if len(dims) == 2:
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    g = ax.scatter(phi[:,dims[0]], phi[:,dims[1]], s=5, c=data_gt)
    cb = plt.colorbar(g)
  else:
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([])
    ax.set_yticks([])
    g = ax.scatter(phi[:,dims[0]], phi[:,dims[1]], phi[:,dims[2]], s=5, c=data_gt)
    cb = plt.colorbar(g)
  if filename:
    plt.savefig(filename)
  plt.show()

def plot_mixture_correlations(mixtures, phi, Sigma, steps, n_comps=2, filename=None):
  start, end, skip = steps
  num_plots = (end - start) // skip
  fig = plt.figure(figsize=(3 * num_plots, 2))
  plt.rc('xtick', labelsize=12)
  plt.rc('ytick', labelsize=12)
  for i,index in enumerate(mixtures[start:end:skip]):
    ax = fig.add_subplot(1, num_plots, i + 1)
    corrs = []
    eigs = []
    v = phi[:,index]
    lambda_v = Sigma[index]
    for m in range(2, n_comps + 1):
      for combo in list(combinations(np.arange(1, index), m)):
        combo = list(combo)
        lambda_sum = np.sum(Sigma[combo])
        lambda_diff = lambda_v - lambda_sum
        eigs.append(lambda_diff)

        # get product of proposed base eigenvectors
        v_combo = np.ones(phi.shape[0])
        for i in combo:
          v_combo *= phi[:,i]

        # test with positive
        corr = syn.calculate_corr(v_combo, v)

        # test with negative
        dcorr = syn.calculate_corr(v_combo, -v)
        corr = max(corr, dcorr)
        corrs.append(corr)

    best_corr = np.max(corrs)
    best_eig_err = eigs[np.argmax(corrs)]
    ax.annotate('{}'.format(np.around(best_corr, 2)),
                 xy=(best_corr, best_eig_err),
                 xytext=(best_corr - 0.1, 0.75 * np.max(eigs)),
                 fontsize=12,
                 arrowprops=dict(arrowstyle='->',
                            color='r'))
    ax.set_title(i)
    ax.set_xlim(0, 1)
    ax.get_xaxis().set_ticks([])
    fig.text(0, 0.5, r'$\lambda_i + \lambda_j - \lambda_k$',
             va='center', rotation='vertical')
    ax.scatter(corrs, eigs, s=3)
  plt.tight_layout(pad=0.5)
  if filename:
    plt.savefig(filename)
  plt.show()

def plot_C_matrix(manifolds, C, filename=None):
  C1 = np.empty((C.shape[0], 0))
  idxs = []
  for m in manifolds:
    idxs.extend(m)
    C1 = np.concatenate((C1, C[:,m]), axis=1)

  C2 = np.empty((0, C1.shape[1]))
  for m in manifolds:
    C2 = np.concatenate((C2, C1[m,:]), axis=0)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xticklabels(['']) # + idxs)
  ax.set_yticklabels(['']) # + idxs)
  matrix = ax.matshow(C2, cmap=plt.cm.Reds)
  fig.colorbar(matrix)
  if filename:
    plt.savefig(filename)
  plt.show()

def plot_k_cut(labels,n_comps, theta, z):
  plt.figure()
  for i in range(len(theta)):
    plt.plot([0, np.cos(theta[i])], [0, np.sin(theta[i])], c='red')
    plt.annotate(labels[0][i], xy=(np.cos(theta[i]), np.sin(theta[i])))
  for j in range(n_comps):
    z_angle = (z + j * 2 * np.pi / n_comps) % (2 * np.pi)
    plt.plot([0, np.cos(z_angle)], [0, np.sin(z_angle)], c='black')
  plt.xlim((-1.1, 1.1))
  plt.ylim((-1.1, 1.1))
  plt.axis('equal')
  plt.show()
