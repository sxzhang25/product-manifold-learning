import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_eigenvectors(data, eigenvectors, labels=None, title=None, filename=None):
  '''
  plots the array eigenvectors

  eigenvectors: a list of eigenvectors
  '''
  rows = int(np.ceil(len(eigenvectors)**0.5))
  cols = rows
  if data.shape[1] <= 2:
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

def plot_correlations(corrs, thresh=None):
  plt.bar(np.arange(len(corrs.values())), sorted(corrs.values()), 2, color='g')
  if thresh:
    plt.plot([0, len(corrs.values())], [thresh, thresh], "k--")
    plt.annotate('thresh = {}'.format(thresh), xy=(0, thresh + 0.01))
  plt.title('Correlations')
  plt.show()

def plot_embedding(phi, dims):
  if len(dims) is 0:
    return
  fig = plt.figure()
  if len(dims) == 2:
    ax = fig.add_subplot(111)
    ax.scatter(phi[:,dims[0]], phi[:,dims[1]], s=5)
  else:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(phi[:,dims[0]], phi[:,dims[1]], phi[:,dims[2]], s=5)
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
