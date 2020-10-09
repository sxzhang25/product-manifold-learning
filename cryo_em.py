import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

import synthetic as syn
from fakekv import *
from plots import *

###
# cryo_em.py
#
# Generates synthetic cryo-EM data.
###

def generate_cryo_em_data(n_samples, x=10, y=10, var=1, seed=0):
  '''
  Generates synthetic cryo-EM data using the FakeKV class. The synthetic
  molecule has two independent conformational components: a rotation and a
  translation.

  n_samples: the number of images to generate
  x: the largest possible translation in either x-direction
  y: the largest possible translation in either y-direction
  var: the variance of gaussian noise added
  seed: the random seed
  '''
  np.random.seed(seed)
  image_data = np.zeros((n_samples, L, L))
  raw_data = np.zeros((n_samples, 3))
  mol = FakeKV()
  for i in range(n_samples):
    # show progress
    if (i % 10 == 0):
      sys.stdout.write('\r')
      sys.stdout.write("[%-20s] %d%%" % ('='*int(20*i/n_samples), 100*i/n_samples))
      sys.stdout.flush()

    angle = 90 * np.random.random()
    shift_x = x * (2 * np.random.random() - 1)
    shift_y = y * (2 * np.random.random() - 1)
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

def downsample_data(data, l):
  '''
  downsamples data with resolution L * L to resolution l * l
  '''
  stride = L // l
  lowres_data = data[:,::stride,::stride]
  return lowres_data
