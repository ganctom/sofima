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
import time
from typing import Sequence, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.io import imread, imsave
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import gaussian_filter
import cv2


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
  """Detect smearing in the image using cross-correlation between line segments.

  Args:
    img: The image array
    segment_width: Length of the line segment(s) used for cross-correlation
    dx: stride in x-axis of the image; defines the 'resolution' of
      the smearing map in X.
    dy: distance between lines to be correlated in vertical
      direction (in pixels). Must be > 1.
    sigma: Standard deviation for Gaussian blur. Do not use sigma=1.0

  Returns:
    smearing_mask: A binary mask indicating the smeared portions.

  """

  # Initialize the smearing mask with zeros (no smearing)
  smearing_map = np.full(np.shape(img), np.nan)

  # Blur the image to speed up the computation
  img = gaussian(img, sigma=sigma) if sigma > 1 else img.astype(float)

  # Pad the entire image to handle borders
  hw = int(segment_width // 2)
  pad_width = ((0, 0), (hw, hw))
  img_padded = np.pad(img, pad_width, mode='constant')

  h, w = np.shape(img)
  for y in range(h - dy):
    for x in range(0, w, dx):

      sec_a = img_padded[y, x:x + segment_width]
      sec_b = img_padded[y + dy, x:x + segment_width]
      corr = signal.correlate(sec_a, sec_b)

      # Find the shift from peak of cross-correlation &
      # calculate the center position of the corr line
      center = len(sec_a)
      peak_shift = np.argmax(corr) - (center - 1)
      smearing_map[y][x] = int(peak_shift)

  return smearing_map


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
  plt.close(fig)
  return


def _create_mask(arr, threshold):
  return np.abs(arr) > threshold

def create_mask(arr, threshold):
  return -threshold > arr


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


def interpolate_nan_2d(arr, min_interp_pts=10):
  """Performs lin. interpolation to replace NaN values in input array along each row.
  Args:
    arr: Input 2D array with NaN values
    min_interp_pts:

  Returns:
    Output 2D array with NaN values replaced by interpolated values.
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


def flood_smearing(mask: np.ndarray, portion: float) -> np.ndarray:
  """

  """

  rows, cols = mask.shape
  win = cols // 3
  result_array = mask.copy()

  for row in range(rows):
    window_sum = np.convolve(mask[row, :], np.ones(win), mode='valid')
    target_sum = portion * win

    condition_met = (window_sum == target_sum)
    if any(condition_met):
      start_index = np.argmax(condition_met)
      # result_array[row, start_index:start_index+win] = True
      result_array[row, start_index:] = True

  return result_array


def remove_isolated(mask: np.ndarray[bool],
                    min_size=300
                    ) -> np.ndarray[bool]:
  """Filters small areas with True value in the 2D-binary mask

  Remove isolated islands of True values in binary mask by finding their
  contours and thresholding their area.

  Args:
    mask: Input binary mask to be filtered
    min_size: Objects smaller than this value (pixels) will be filtered

  Returns:
    Filtered binary mask array
  """

  mask_uint8 = mask.astype(np.uint8)
  contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

  for c in contours:
    if cv2.contourArea(c) < min_size:
      kwargs = {'contours': [c],
                'contourIdx': -1,
                'color': 0,
                'thickness': cv2.FILLED
                }
      cv2.drawContours(mask_uint8, **kwargs)

  return mask_uint8 > 0


def get_smearing_mask(
  img: np.ndarray,
  mask_top_edge: int = 0,
  path_plot: Optional[str] = None,
  plot=False
) -> Optional[np.ndarray]:
  """Computes mask of a distortion appearing at the top of the EM-images.

  Estimate the presence and extent of a smearing distortion at the top of the
  input image and return it as a boolean mask.

  Args:
    img: input image for detection of distortion at its top border
    mask_top_edge: number of lines at the top of the image to be masked entirely
    path_plot: filepath where to save the mask image (if plot=True)
    plot: switch to execute creation of various mask graphs

  Returns:
    Mask of smearing distortion with the shape same as the input image
  """

  det_args = dict(
    img=img,
    segment_width=1000,
    sigma=1.5,
    dx=50,
    dy=4
  )

  # Run smearing detection
  smr_map = detect_smearing2d(**det_args)
  smr_map_interp = interpolate_nan_2d(smr_map)
  smr_map_interp = gaussian(smr_map_interp, sigma=2)
  mask = create_mask(smr_map_interp, threshold=0.1)

  clean_args = dict(
    mask=mask,
    min_size=800,
    portion=1.0,
    max_vert_extent=450,
    top=mask_top_edge
  )

  def clean_mask(mask, top, min_size, portion, max_vert_extent):

    # Mask entire top lines
    if top > 0:
      mask[:top] = True

    # Mask top right border # TODO investigate if needed
    mask[:, -1] = True

    # Fill binary holes
    mask = fill_holes(mask)

    # Unmask all lines below line nr. 'max_vert_extent'
    if 0 < max_vert_extent < mask.shape[0]:
      mask[max_vert_extent:] = False

    # Mask small masking irregularities
    if portion > 0:
      mask = flood_smearing(mask, portion)

    # Remove True islands with small area
    if min_size > 0:
      mask = remove_isolated(mask, min_size)

    return mask

  mask_final2 = clean_mask(**clean_args)


  # mask_filled = fill_holes(mask)
  # # mask_flooded = flood_pixels(mask_filled, ratio=0.4)
  # # mask_flood = flood_smearing(mask_flooded, portion=1.0)
  # mask_flood = flood_smearing(mask_filled, portion=1.0)
  # # mask_flood2 = flood_smearing(mask_flood, portion=.5)
  # # mask_final = fill_holes(mask_flood2)
  # mask_final = fill_holes(mask_flood)
  # mask_final2 = remove_isolated(mask_final, min_size=800)

  # if plot:
  #   to_plot = [smr_map, smr_map_interp, mask_flood,
  #              mask_flood, mask_final, mask_final2
  #              ]
  #   plot_smearing(plots=to_plot, path_plot=path_plot)

  return mask_final2
