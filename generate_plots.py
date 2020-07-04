import time
import sys
import json

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###
# PLOT DATA
###

def plot_og_data(data, title=None, filename=None):
  '''
  plots the original data
  '''

  fig = plt.figure(figsize=(5,5))
  if data.shape[1] <= 2:
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0], data[:,1], s=5)
  elif data.shape[1] == 3:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], s=5)
  else:
    ax = fig.add_subplot(111, projection='3d')
    g = ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3], s=5)
    cb = plt.colorbar(g)
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
        ax = axs[r, c]
        index = r * cols + c
        if index >= len(eigenvectors):
          ax.set_visible(False)
        else:
          v = eigenvectors[r * cols + c]
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

  plt.show()


def main():
  datafile = sys.argv[1]
  with open(datafile) as f:
    params = json.load(f)

  # unpack parameters
  print('Parameters...')
  print(params)

  name = params['name']
  test_name = params['test_name']
  precomputed = params['precomputed']
  l1 = params['l1']
  l2 = np.sqrt(np.pi) + params['l2']
  n_samples = params['n_samples']
  seed = params['seed']
  datatype = params['datatype']
  sigma = params['sigma']
  n_comps = params['n_comps']
  n_eigenvectors = params['n_eigenvectors']
  K = params['K']
  dist_thresh = params['dist_thresh']

  # load data
  data_filename = './data/data_' + name + '.dat'
  phi_filename = './data/phi_' + name + '.dat'

  data = np.loadtxt(data_filename)
  l1, l2 = data[0,:2]
  data = data[1:,:]
  phi = np.loadtxt(phi_filename)

  if datatype == 'line_circle':
    data_r = np.zeros((data.shape[0],2))
    data_r[:,0] = data[:,0]
    data_r[:,1] = np.arctan2(data[:,2], data[:,1])
    data = data_r
  elif datatype == 'rect_circle':
    data_r = np.zeros((data.shape[0],3))
    data_r[:,0] = data[:,0]
    data_r[:,1] = data[:,1]
    data_r[:,2] = np.arctan2(data[:,3], data[:,2])
    data = data_r

  if not precomputed:
    # plot original data
    plot_og_data(data, title='Original Data ({})'.format(name), filename='./images/{}_{}_original_data.png'.format(name, test_name))

    # plot eigenvectors
    vecs = [phi[:,i] for i in range(1, n_eigenvectors+1)]
    eigenvectors_filename = './images/' + name + '_' + test_name + '_eigenvalues_' + str(n_eigenvectors) + '.png'
    plot_eigenvectors(data,
                      vecs[:20], # CHANGE BACK TO 100
                      labels=[int(i) for i in range(1,n_eigenvectors+1)],
                      title='Laplace Eigenvectors ({})'.format(name),
                      filename=eigenvectors_filename)

  # plot best matches
  manifold1 = np.loadtxt('./data/manifold1_{}_{}.dat'.format(name, test_name))
  manifold2 = np.loadtxt('./data/manifold2_{}_{}.dat'.format(name, test_name))

  vecs1 = [phi[:,int(i)] for i in manifold1]
  vecs2 = [phi[:,int(i)] for i in manifold2]

  plot_eigenvectors(data,
                    vecs1,
                    labels=[int(i) for i in manifold1],
                    filename='./images/manifold1_{}_{}.png'.format(name, test_name))

  plot_eigenvectors(data,
                    vecs2,
                    labels=[int(i) for i in manifold2],
                    filename='./images/manifold2_{}_{}.png'.format(name, test_name))

  plot_independent_eigenvectors(manifold1,
                                manifold2,
                                n_eigenvectors,
                                title='d={}'.format(dist_thresh),
                                filename='./images/{}_{}_{}_{}_eigenvector_division.png'.format(name, test_name, K, dist_thresh))

if __name__ == '__main__':
  main()
