'''
merge in local region
'''
import numpy as np

def merge_2by2(prob):
  '''
  merge 2x2 for affinity
  '''
  img_height = prob.shape[0]
  img_width = prob.shape[1]
  num_stride = prob.shape[2]
  num_neighbor = prob.shape[3]
  if img_height % 2 != 0:
    img_height = img_height - 1
    prob = prob[:-1, :, :, :]
  if img_width % 2 != 0:
    img_width = img_width - 1
    prob = prob[:, :-1, :, :]

  new_prob = np.zeros(
      [img_height//2, img_width//2, num_stride-1, num_neighbor])

  new_prob = prob[::2, ::2, 1:, :] + prob[::2, 1::2, 1:, :] + \
      prob[1::2, ::2, 1:, :] + prob[1::2, 1::2, 1:, :]
  new_prob = new_prob / 4

  new_prob[:, :, 0, 0] = (new_prob[:, :, 0, 0]*4 + prob[::2, ::2, 0, 0]) / 5
  new_prob[:, :, 0, 2] = (new_prob[:, :, 0, 2]*4 + prob[::2, 1::2, 0, 2]) / 5
  new_prob[:, :, 0, 5] = (new_prob[:, :, 0, 5]*4 + prob[1::2, ::2, 0, 5]) / 5
  new_prob[:, :, 0, 7] = (new_prob[:, :, 0, 7]*4 + prob[1::2, 1::2, 0, 7]) / 5
  new_prob[:, :, 0, 1] = (new_prob[:, :, 0, 1]*4 + prob[::2, ::2, 0, 1] +
                          prob[::2, ::2, 0, 2]+prob[::2, 1::2, 0, 1]+prob[::2, 1::2, 0, 0]) / 8
  new_prob[:, :, 0, 3] = (new_prob[:, :, 0, 3]*4 + prob[::2, ::2, 0, 3] +
                          prob[::2, ::2, 0, 5]+prob[1::2, ::2, 0, 3]+prob[1::2, ::2, 0, 0]) / 8
  new_prob[:, :, 0, 4] = (new_prob[:, :, 0, 4]*4 + prob[::2, 1::2, 0, 4] +
                          prob[::2, 1::2, 0, 7]+prob[1::2, 1::2, 0, 4]+prob[1::2, 1::2, 0, 2]) / 8
  new_prob[:, :, 0, 6] = (new_prob[:, :, 0, 6]*4 + prob[1::2, ::2, 0, 6] +
                          prob[1::2, ::2, 0, 7]+prob[1::2, 1::2, 0, 6]+prob[1::2, 1::2, 0, 5]) / 8

  return new_prob


def merge_semantic_2by2(prob):
  '''
  merge 2x2 for semantic
  '''
  img_height = prob.shape[0]
  img_width = prob.shape[1]
  if img_height % 2 != 0:
    img_height = img_height - 1
    prob = prob[:-1, :, :]
  if img_width % 2 != 0:
    img_width = img_width - 1
    prob = prob[:, :-1, :]

  merged_prob = np.zeros(
      (img_height // 2, img_width // 2, prob.shape[2]), dtype=np.float32)
  merged_prob = (prob[::2, ::2, :] + prob[::2, 1::2, :] +
                 prob[1::2, ::2, :] + prob[1::2, 1::2, :]) / 4

  return merged_prob


def merge_4by4(prob):
  '''
  merge 4x4 for affinity
  '''
  img_height = prob.shape[0]
  img_width = prob.shape[1]
  num_stride = prob.shape[2]
  num_neighbor = prob.shape[3]
  if img_height % 4 != 0:
    img_height = img_height // 4 * 4
    prob = prob[:img_height, :, :, :]
  if img_width % 4 != 0:
    img_width = img_width // 4 * 4
    prob = prob[:, :img_width, :, :]
  prob_2by2 = merge_2by2(prob)
  new_prob = np.zeros(
      [img_height//4, img_width//4, num_stride-2, num_neighbor])

  new_prob = prob_2by2[::2, ::2, 1:, :] + prob_2by2[::2, 1::2, 1:, :] + \
             prob_2by2[1::2, ::2, 1:, :] + prob_2by2[1::2, 1::2, 1:, :]
  new_prob = new_prob / 4

  new_prob[:, :, 0, 0] = (new_prob[:, :, 0, 0]*16 +
                          prob_2by2[::2, ::2, 0, 0]*5) / 21
  new_prob[:, :, 0, 2] = (new_prob[:, :, 0, 2]*16 +
                          prob_2by2[::2, 1::2, 0, 2]*5) / 21
  new_prob[:, :, 0, 5] = (new_prob[:, :, 0, 5]*16 +
                          prob_2by2[1::2, ::2, 0, 5]*5) / 21
  new_prob[:, :, 0, 7] = (new_prob[:, :, 0, 7]*16 +
                          prob_2by2[1::2, 1::2, 0, 7]*5) / 21
  new_prob[:, :, 0, 1] = (new_prob[:, :, 0, 1]*16 +
                          prob_2by2[::2, ::2, 0, 1]*8 +
                          prob_2by2[::2, ::2, 0, 2]*5 +
                          prob_2by2[::2, 1::2, 0, 1]*8 +
                          prob_2by2[::2, 1::2, 0, 0]*5) / 42
  new_prob[:, :, 0, 3] = (new_prob[:, :, 0, 3]*16 +
                          prob_2by2[::2, ::2, 0, 3]*8 +
                          prob_2by2[::2, ::2, 0, 5]*5 +
                          prob_2by2[1::2, ::2, 0, 3]*8 +
                          prob_2by2[1::2, ::2, 0, 0]*5) / 42
  new_prob[:, :, 0, 4] = (new_prob[:, :, 0, 4]*16 +
                          prob_2by2[::2, 1::2, 0, 4]*8 +
                          prob_2by2[::2, 1::2, 0, 7]*5 +
                          prob_2by2[1::2, 1::2, 0, 4]*8 +
                          prob_2by2[1::2, 1::2, 0, 2]*5) / 42
  new_prob[:, :, 0, 6] = (new_prob[:, :, 0, 6]*16 +
                          prob_2by2[1::2, ::2, 0, 6]*8 +
                          prob_2by2[1::2, ::2, 0, 7]*5 +
                          prob_2by2[1::2, 1::2, 0, 6]*8 +
                          prob_2by2[1::2, 1::2, 0, 5]*5) / 42

  return new_prob


def merge_semantic_4by4(prob):
  '''
  merge 4x4 for semantic
  '''
  img_height = prob.shape[0]
  img_width = prob.shape[1]
  if img_height % 4 != 0:
    img_height = img_height // 4 * 4
    prob = prob[:img_height, :, :]
  if img_width % 4 != 0:
    img_width = img_width // 4 * 4
    prob = prob[:, :img_width, :]

  merged_prob = np.zeros(
      (img_height // 4, img_width // 4, prob.shape[2]), dtype=np.float32)
  for i in range(4):
    for j in range(4):
      merged_prob += prob[i::4, j::4, :]
  merged_prob = merged_prob / 16

  return merged_prob
