# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for manipulating flow arrays.

A flow field has the same physical representation as a relative coordinate map
(see map_utils.py). Flow vectors can have additional statistics associated
with them. When present, these are stored in channels 2+ of the array.

Flow entries can be invalid (i.e., unknown for a given point), in which
case they are marked by nan stored in both X and Y channels.
"""

from typing import Sequence, List, Optional

import numpy as np
from scipy import ndimage

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.io import imread, imsave
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import binary_fill_holes


def apply_mask(flow, mask):
  for i in range(flow.shape[0]):
    flow[i, ...][mask] = np.nan


def clean_flow(flow: np.ndarray, min_peak_ratio: float,
               min_peak_sharpness: float, max_magnitude: float,
               max_deviation: float, dim: int = 2) -> np.ndarray:
  """Removes flow vectors that do not fulfill quality requirements.

  Args:
    flow: [c, z, y, x] flow field
    min_peak_ratio: min. value of peak intensity ratio (chan 3); only
      applies to 4-channel input flows
    min_peak_sharpness: min. value of the peak sharpness (chan 2); only
      applies to 4-channel input flows
    max_magnitude: maximum magnitude of a flow component; when <= 0,
      the constraint is not applied
    max_deviation: maximum absolute deviation from the 3x3-window median of a
      flow component; when <= 0, the constraint is not applied
    dim: number of spatial dimensions of the flow field

  Returns:
    filtered flow field in a [2 or 3, z, y, x] array; 3 output channels
    are used only when the input array also has c=3
  """
  assert dim in (2, 3)
  assert dim <= flow.shape[0] <= dim + 2
  if flow.shape[0] == dim + 2:
    ret = flow[:dim, ...].copy()
    bad = np.abs(flow[dim, ...]) < min_peak_sharpness
    pr = np.abs(flow[dim + 1, ...])
    bad |= (pr > 0.0) & (pr < min_peak_ratio)
  else:
    ret = flow.copy()
    bad = np.zeros(flow[0, ...].shape, dtype=bool)

  if max_magnitude > 0:
    bad |= np.max(np.abs(flow[:dim, ...]), axis=0) > max_magnitude

  if max_deviation > 0:
    size = (1, 1, 3, 3) if dim == 2 else (1, 3, 3, 3)
    med = ndimage.median_filter(np.nan_to_num(flow[:dim, ...]), size=size)
    bad |= (np.max(np.abs(med - flow[:dim, ...]), axis=0) > max_deviation)

  apply_mask(ret, bad)
  return ret


def reconcile_flows(flows: Sequence[np.ndarray], max_gradient: float,
                    max_deviation: float, min_patch_size: int,
                    min_delta_z: int = 0) -> np.ndarray:
  """Reconciles multiple flows.

  Args:
    flows: sequence of [c, z, y, x] flow arrays, sorted in order of decreasing
      preference; 'c' can be 2 or 3
    max_gradient: maximum absolute value of the gradient of a flow component;
      when <= 0, the constraint is not applied
    max_deviation: maximum absolute deviation from the 3x3-window median of a
      flow component; when <= 0, the constraint is not applied
    min_patch_size: minimum size of a connected component of the flow field
      in pixels; when <= 0, the constraint is not applied
    min_delta_z: for 3-channel flows, the minimum absolute value of the z
      offset at which flow data is considered valid

  Returns:
    reconciled flow field in a [c, z, y, x] array
  """
  ret = flows[0].copy()
  assert ret.shape[0] in (2, 3)
  for _, f in enumerate(flows[1:]):
    # Try to fill any invalid values.
    m = np.repeat(np.isnan(ret[0:1, ...]), ret.shape[0], 0)
    if ret.shape[0] == 3:
      m &= np.repeat(np.abs(f[2:3, ...]) >= min_delta_z, 3, 0)
    ret[m] = f[m]

  if max_gradient > 0:
    # Invalidate regions where the gradient is too large.
    m = np.abs(np.diff(ret[0, ...], axis=-1, prepend=0)) > max_gradient
    m |= np.abs(np.diff(ret[0, ...], axis=-1, append=0)) > max_gradient
    m |= np.abs(np.diff(ret[1, ...], axis=-2, prepend=0)) > max_gradient
    m |= np.abs(np.diff(ret[1, ...], axis=-2, append=0)) > max_gradient
    apply_mask(ret, m)

  # Filter out points that deviate too much from the median. This gets rid
  # of small, few-point anomalies.
  if max_deviation > 0:
    med = ndimage.median_filter(np.nan_to_num(ret), size=(1, 1, 3, 3))
    bad = (np.max(np.abs(med - ret)[:2, ...], axis=0) > max_deviation)
    apply_mask(ret, bad)

  if min_patch_size > 0:
    bad = np.zeros(ret[0, ...].shape, dtype=bool)
    valid = ~np.any(np.isnan(ret), axis=0)
    for z in range(valid.shape[0]):
      labeled, _ = ndimage.label(valid[z, ...])
      ids, sizes = np.unique(labeled, return_counts=True)
      small = ids[sizes < min_patch_size]
      bad[z, ...][np.in1d(labeled.ravel(), small).reshape(labeled.shape)] = True
    apply_mask(ret, bad)

  return ret

def detect_smearing2d(
  img: np.ndarray,
  segment_width: int = 1000,
  dx: int = 10,
  dy: int = 3,
  sigma=1.5
) -> Optional[np.ndarray]:
  """
  Detect smearing in the image using cross-correlation between line segments.

  Args:
  - img (numpy.ndarray): The image array.
  - segment_width (int): Length of the line segment(s) used for cross-correlation.
  - dx (int): stride in x-axis of the image; defines the 'resolution' of the smearing map in X.
  - dy (int): distance between lines to be correlated in vertical direction (in pixels). Must be > 1.
  - sigma (float): Standard deviation for Gaussian blur. Do not use sigma=1.0

  Returns:
  - smearing_mask (numpy.ndarray): A binary mask indicating the smeared portions.

  """

  # Initialize the smearing mask with zeros (no smearing)
  smearing_map = np.full(np.shape(img), np.nan)

  # Blur the image to speed up the computation
  img = gaussian(img, sigma=sigma) if sigma > 1 else img.astype(float)

  # Pad the entire image to handle borders
  half_width = int(segment_width // 2)
  mode = 'constant'
  pad_width = ((0, 0), (half_width, half_width))
  img_padded = np.pad(img, pad_width, mode)

  # Iterate over the line segments
  for i in range(np.shape(img)[0] - dy):
    for j in range(0, np.shape(img)[1], dx):
      # Perform cross-correlation between neighboring line sections
      sec_a = img_padded[i, j:j + segment_width]
      sec_b = img_padded[i + dy, j:j + segment_width]
      corr = signal.correlate(sec_a, sec_b)
      # Find the peak of the cross-correlation &
      # Calculate the center position of the line
      peak = np.argmax(corr)
      center = len(sec_a)
      # Calculate the peak shift
      peak_shift = peak - (center - 1)
      smearing_map[i][j] = int(peak_shift)
  return smearing_map


def create_mask(arr, threshold):
  return np.abs(arr) > threshold

def fill_holes(mask):
  num_mask = np.asarray(mask, dtype=int)
  fill_mask = binary_fill_holes(num_mask).astype(bool)
  return fill_mask

def blur_mask(mask, sigma: float = 1.5):
  mask = np.asarray(mask, dtype=bool)
  blurred_mask = gaussian(mask.astype(float), sigma=sigma)
  return np.abs(blurred_mask) > 0


def flood_pixels(mask, ratio=0.75):
  result_mask = mask.copy()

  for i in range(mask.shape[0]):
    row = mask[i, :]
    count = np.sum(row)
    threshold = ratio * len(row)
    if count > threshold:
      result_mask[i, :] = True

  return result_mask


def set_lines_above_recursive(mask, row):
  if row > 0 and np.all(mask[row, :]):
    mask[row - 1, :] = True
    set_lines_above_recursive(mask, row - 1)


def set_lines_above_to_true_recursive(mask):
  result_mask = mask.copy()
  for i in range(1, mask.shape[0]):
    if np.all(mask[i, :]):
      result_mask[i - 1, :] = True
      set_lines_above_recursive(result_mask, i - 1)
  return result_mask


def plot_smearing(plots: List[np.ndarray], path_plot: str) -> None:
  fig = plt.figure(figsize=(15, 8))
  gs = gridspec.GridSpec(7, 2, width_ratios=[1, 0.1])

  axes = [plt.subplot(gs[i, 0]) for i in range(6)]
  cax = plt.subplot(gs[:, 1])

  im0 = axes[0].matshow(plots[0], cmap='viridis')
  axes[0].set_title('Original cross-correlation map')

  im1 = axes[1].matshow(plots[1], cmap='viridis')
  axes[1].set_title('Interpolated cross-correlation map')

  im2 = axes[2].imshow(plots[2], cmap='gray')
  axes[2].set_title('Mask: Interpolated cross-correlation map')

  im3 = axes[3].imshow(plots[3], cmap='gray')
  axes[3].set_title('Mask: Filled holes')

  im4 = axes[4].imshow(plots[4], cmap='gray')
  axes[4].set_title('Mask: Filled holes + Flooded')

  im5 = axes[5].imshow(plots[5], cmap='gray')
  axes[5].set_title('Mask: Filled holes + Flooded (0.7 threshold) + LineFill')
  axes[5].set_ylabel('Line nr.')
  axes[5].set_xlabel('Column nr.')

  fig.colorbar(im0, cax=cax, orientation='vertical')
  plt.tight_layout()
  plt.savefig(path_plot)
  # plt.show()
  return


def interpolate_nan_2d(arr, min_interp_pts=10):
  """
  Perform linear interpolation to replace NaN values in a 2D array along each row.
  Parameters:
  - array_2d (numpy.ndarray): Input 2D array with NaN values.
  Returns:
  - interpolated_array (numpy.ndarray): Output 2D array with NaN values replaced by interpolated values.
  """

  interp_arr = arr.copy()
  for i, row in enumerate(arr):
    if np.isnan(row).all() or np.sum(~np.isnan(row)) < min_interp_pts:
      continue
    nan_idx = np.isnan(row)
    indices = np.arange(len(row))
    interp_vals = np.interp(indices, indices[~nan_idx], row[~nan_idx])
    interp_arr[i, nan_idx] = interp_vals[nan_idx]
  return interp_arr


def get_smearing_mask(
  img: np.ndarray,
  smr_ext: int = 4,
  path_plot: Optional[str] = None,
  plot=False
) -> Optional[np.ndarray]:

  kwargs = dict(
    img=img,
    segment_width=1000,
    sigma=1.5,
    dx=10,
    dy=3
  )

  # Apply the smearing detection
  smr_map = detect_smearing2d(**kwargs)
  smr_map_interp = interpolate_nan_2d(smr_map)
  smr_map_interp = gaussian(smr_map_interp, sigma=2)
  mask_smr = create_mask(smr_map_interp, threshold=0.1)
  mask_filled = fill_holes(mask_smr)
  mask_flooded = flood_pixels(mask_filled, ratio=0.7)
  # mask_final = set_lines_above_to_true_recursive(mask_filled)
  # mask_final = set_lines_above_to_true_recursive(mask_flooded)
  mask_final = mask_filled

  if smr_ext > 0:
    mask_final[:smr_ext] = True

  if plot:
    to_plot = [smr_map, smr_map_interp, mask_smr, mask_filled, mask_flooded,
               mask_final]
    plot_smearing(plots=to_plot, path_plot=path_plot)

  return mask_final
