import time
import sys
import json
import pickle

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

  plt.show()


def main():
  datafile = sys.argv[1]
  with open(datafile) as f:
    params = json.load(f)

  # unpack parameters
  print('Parameters...')
  print(params)

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

  # load pickle file containing all info
  with open('./data/{}_info.pickle'.format(test_name), 'rb') as handle:
    info = pickle.load(handle)
  data = info['data']
  phi = info['phi']


  if datatype == 'line_circle':
    data_r = np.zeros((data.shape[0],2))
    data_r[:,0] = data[:,0]
    data_r[:,1] = np.arctan2(data[:,2], data[:,1])
  elif datatype == 'rect_circle':
    data_r = np.zeros((data.shape[0],3))
    data_r[:,0] = data[:,0]
    data_r[:,1] = data[:,1]
    data_r[:,2] = np.arctan2(data[:,3], data[:,2])
  else:
    data_r = data

  if not precomputed:
    # plot original data
    plot_og_data(data, title='Original Data', filename='./images/{}_original_data.png'.format(test_name))

    # plot eigenvectors
    vecs = [phi[:,i] for i in range(n_eigenvectors)]
    eigenvectors_filename = './images/' + test_name + '_eigenvalues_' + str(n_eigenvectors) + '.png'
    plot_eigenvectors(data_r,
                      vecs[:100],
                      labels=[int(i) for i in range(n_eigenvectors)],
                      title='Laplace Eigenvectors',
                      filename=eigenvectors_filename)

  # plot best matches
  manifold1 = info['manifold1']
  manifold2 = info['manifold2']

  vecs1 = [phi[:,int(i)] for i in manifold1]
  vecs2 = [phi[:,int(i)] for i in manifold2]

  plot_eigenvectors(data_r,
                    vecs1,
                    labels=[int(i) for i in manifold1],
                    filename='./images/manifold1_{}.png'.format(test_name))

  plot_eigenvectors(data_r,
                    vecs2,
                    labels=[int(i) for i in manifold2],
                    filename='./images/manifold2_{}.png'.format(test_name))

  plot_independent_eigenvectors(manifold1,
                                manifold2,
                                n_eigenvectors,
                                title='manifold split',
                                filename='./images/{}_{}_eigenvector_division.png'.format(test_name, K))

if __name__ == '__main__':
  main()
