# generate datasets used in the papers.

import time
import sys
import json
import pickle
import os
import argparse
import numpy as np

from fakekv import *

def generate_synthetic_data(dimensions, n_samples, datatype='rectangle',
                            seed=0, noise=0.05):
  '''
  generates uniform random data from simple geometric manifolds
  dimensions: the dimensions of the data
  num_samples: the number of samples to generate
  seed: the random seed for generating the data
  type: the type of data to generate
  '''

  np.random.seed(seed)

  if datatype=='rectangle':
    # rectangle
    l1, l2 = dimensions
    line1_data = l1 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line2_data = l2 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    data = np.column_stack((line1_data, line2_data))

  elif datatype=='rectangle3d':
    # rectangle
    l1, l2, z_noise = dimensions
    line1_data = l1 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line2_data = l2 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line3_data = np.random.normal(scale=z_noise, size=n_samples)
    data = np.column_stack((line1_data, line2_data, line3_data))

  elif datatype=='line_circle':
    # hollow cylinder
    l1, l2 = dimensions
    line_data = l1 * (np.random.rand(n_samples) +  \
                      np.random.normal(scale=noise, size=n_samples))
    circle_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle_data[:,0] = (l2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle_data[:,1] = (l2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    data = np.column_stack((line_data, circle_data))

  elif datatype=='cube':
    # cube
    l1, l2, l3 = dimensions
    line1_data = l1 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line2_data = l2 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    line3_data = l3 * (np.random.rand(n_samples) +  \
                       np.random.normal(scale=noise, size=n_samples))
    data = np.column_stack((line1_data, line2_data, line3_data))

  elif datatype=='torus':
    #torus
    r1, r2 = dimensions
    circle1_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle1_data[:,0] = r1 + (r1 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle1_data[:,1] = r1 + (r1 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    circle2_data = np.empty((n_samples,2))
    theta = 2 * np.pi * np.random.rand(n_samples)
    circle2_data[:,0] = r2 + (r2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.cos(theta)
    circle2_data[:,1] = r2 + (r2 + noise * np.random.normal(scale=noise, size=n_samples)) * np.sin(theta)
    data = np.column_stack((circle1_data, circle2_data))

  else:
    print('Error: invalid data type')
    return

  return data

def generate_cryo_em_data(n_samples, x_stretch=0, y_stretch=0, var=1, seed=0):
  '''
  Generates synthetic cryo-EM data using the FakeKV class. The synthetic
  molecule has two independent conformational components: a rotation and a
  stretch.

  n_samples: the number of images to generate
  x: the largest possible stretch in either x-direction
  y: the largest possible stretch in either y-direction
  var: the variance of gaussian noise added
  seed: the random seed
  '''
  np.random.seed(seed)
  image_data = np.zeros((n_samples, L, L))
  raw_data = np.zeros((n_samples, 3))
  mol = FakeKV()
  for i in range(n_samples):
    # show progress in generating data
    if (i % 10 == 0):
      sys.stdout.write('\r')
      sys.stdout.write("[%-20s] %d%%" % ('='*int(20*i/n_samples), 100*i/n_samples))
      sys.stdout.flush()

    angle = 90 * np.random.random()
    shift_x = x_stretch * (2 * np.random.random() - 1)
    shift_y = y_stretch * (2 * np.random.random() - 1)
    vol = mol.generate(angle=angle, shift=(shift_x, shift_y))
    vol = ndimage.rotate(vol, -30, (0,2), reshape=False, order=1)

    # project vol in direction of z-axis
    projz = np.sum(vol, axis=2, keepdims=False)

    # add gaussian noise
    gauss = np.random.normal(0, var**0.5, (L,L))
    noisy = projz + gauss

    image_data[i,:,:] = noisy
    raw_data[i,:] = [shift_x, shift_y, angle]

  return image_data, raw_data

def main():
  parser = argparse.ArgumentParser(description='Generate data.')
  parser.add_argument("params_files", nargs='+')
  params_files = parser.parse_args().params_files
  
  data_dir = './data/'
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  
  for params_file in params_files:
    print("\nGenerating file:", params_file)
    
    info = {}
    with open(params_file) as f:
      params = json.load(f)
    for item, value in params.items():
      info[item] = value
      print("{:15}:  {}".format(item, value))

    datatype = params['datatype']
    if datatype == "cryo-em":
      name = params['name']
      var = params['var']
      n_samples = params['n_samples']
      seed = params['seed']
      x_stretch = params['x_stretch']
      y_stretch = params['y_stretch']
    
      # generate random data
      print("\nGenerating random cryo-EM data...")
      
      image_data, raw_data = generate_cryo_em_data(n_samples, 
                                                   x_stretch=x_stretch, 
                                                   y_stretch=y_stretch, 
                                                   var=var)
      info['image_data'] = image_data
      info['raw_data'] = raw_data
      
      # save info dictionary using pickle
      print("\nSaving data...")
      data_filename = data_dir + '{}_info.pickle'.format(name)
      with open(data_filename, 'wb') as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print("Done")
      
    else:
      name = params['name']
      dimensions = params['dimensions']
      dimensions[0] = np.sqrt(np.pi) + dimensions[0] # optional
      noise = params['noise']
      n_samples = params['n_samples']
      seed = params['seed']
      
      # generate random data
      print("\nGenerating random data...")
      data = generate_synthetic_data(dimensions,
                                     noise=noise,
                                     n_samples=n_samples,
                                     datatype=datatype,
                                     seed=seed)
      info['data'] = data
      
      # save info dictionary using pickle
      print("\nSaving data...")
      data_filename = data_dir + '{}_info.pickle'.format(name)
      with open(data_filename, 'wb') as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print("Done")

if __name__ == "__main__":
  main()      
  