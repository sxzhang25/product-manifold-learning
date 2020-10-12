# This code was repurposed from from the SPECVOLS repository
# https://github.com/PrincetonUniversity/specvols

import numpy as np
import scipy.ndimage as ndimage
import os
import urllib.request
import urllib.parse
import mrcfile

DATA_URL = 'https://spr.math.princeton.edu/examples/'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/fakekv_data/')
L = 108

class FakeKV:
  def __init__(self):
    _download_fakekv()
    with mrcfile.open(os.path.join(DATA_DIR,'FakeKvMapAlphaOne.mrc')) as mrc_alpha_one:
      self.alpha_one = np.maximum(mrc_alpha_one.data, 0)
    with mrcfile.open(os.path.join(DATA_DIR, 'FakeKvMapAlphaTwo.mrc')) as mrc_alpha_two:
      self.alpha_two = np.maximum(mrc_alpha_two.data, 0)
    with mrcfile.open(os.path.join(DATA_DIR, 'FakeKvMapBeta.mrc')) as mrc_beta:
      self.beta = np.maximum(mrc_beta.data, 0)

    assert self.alpha_one.shape == self.alpha_two.shape == self.beta.shape == (L, L, L)

  def generate(self, angle, shift):
    vol = ndimage.rotate(self.alpha_two, angle, (1,2), reshape=False, order=1) + _stretch_fakekv_middle_bottom(self.alpha_one + self.beta, shift)
    return vol

def _download_fakekv():
  os.makedirs(DATA_DIR, exist_ok=True)
  for name in ['FakeKvMapAlphaOne.mrc', 'FakeKvMapAlphaTwo.mrc', 'FakeKvMapBeta.mrc']:
    data_file = os.path.join(DATA_DIR, name)
    url_name = urllib.parse.urljoin(DATA_URL, name)
    if not os.path.exists(data_file):
      print(f'Downloading file from {url_name}, saving at location {data_file}.')
      urllib.request.urlretrieve(url_name, data_file)

def _stretch_fakekv_middle_bottom(vol, shift):
  (shift_x, shift_y) = shift
  assert vol.shape == (108, 108, 108)
  # Note! We change the entries from Z_START to Z_END, but would copy the other ones over unaltered.
  # So we're doing vol.copy() and then changing the relevant ones rather than initializing to zeroes and then copying over everything as appropriate
  out_vol = vol.copy()

  Z_START = 16
  Z_END = 55

  for z in range(Z_START, Z_END):
    shift_amount = ((Z_END - 1 - z) / (Z_END - Z_START - 1))**2
    out_vol[z - 1,:,:] = ndimage.shift(vol[z - 1,:,:], (shift_y * shift_amount, shift_x * shift_amount), order=1)

  return out_vol

def save_mrc(vol, file_name):
  print(f'Saving to file name {file_name}')
  with mrcfile.new(file_name, overwrite=True) as mrc:
    mrc.set_data(vol)
