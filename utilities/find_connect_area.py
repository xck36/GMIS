'''
find connect area on semantic segmentation result
'''
import numpy as np
from skimage.measure import label
from scipy.ndimage.interpolation import zoom


def find_connect_area(sem_mask, merge_area_size, expend_border=False):
  '''
  find connect area on semantic segmentation result
  '''
  img_width = sem_mask.shape[1]
  img_height = sem_mask.shape[0]

  pad_img_width = (img_width - 1) // merge_area_size * merge_area_size + merge_area_size
  pad_img_height = (img_height - 1) // merge_area_size * merge_area_size + merge_area_size

  pad_sem_mask = np.zeros((pad_img_height, pad_img_width))
  pad_sem_mask[:img_height, :img_width] = sem_mask

  merge_area_width = pad_img_width//merge_area_size
  merge_area_height = pad_img_height//merge_area_size
  tmp = np.zeros((merge_area_height, merge_area_width))
  for i in range(merge_area_size):
    for j in range(merge_area_size):
      tmp = tmp+pad_sem_mask[i::merge_area_size, j::merge_area_size]

  tmp = (tmp > 0).astype('int32')
  mask = label(tmp, 8).astype('uint8')
  if expend_border and merge_area_size > 2:
    tmp_mask = zoom(mask, (2, 2), order=0)

    tmp_mask[1:, :] = tmp_mask[1:, :] + tmp_mask[:-1, :] * \
        ((tmp_mask[1:, :] == 0).astype('int32'))
    tmp_mask[:-1, :] = tmp_mask[:-1, :] + tmp_mask[1:, :] * \
        ((tmp_mask[:-1, :] == 0).astype('int32'))
    tmp_mask[:, 1:] = tmp_mask[:, 1:] + tmp_mask[:, :-1] * \
        ((tmp_mask[:, 1:] == 0).astype('int32'))
    tmp_mask[:, :-1] = tmp_mask[:, :-1] + tmp_mask[:, 1:] * \
        ((tmp_mask[:, :-1] == 0).astype('int32'))
    component_area = zoom(
        tmp_mask, (merge_area_size/2., merge_area_size/2.), order=0)
  else:
    component_area = zoom(mask, (merge_area_size, merge_area_size), order=0)

  return component_area[:img_height, :img_width]
