'''
demo for instance segmentation
'''
import os
import glob
import time
import argparse
import multiprocessing
import warnings
import random
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom

import utilities.show_mask as show_mask
import utilities.get_default_palette as get_default_palette
from utilities.merge import merge_2by2, merge_4by4, merge_semantic_2by2, merge_semantic_4by4
from utilities.find_connect_area import find_connect_area
import gen_evaluation_file


def parse_args():
  '''
  input parameters
  '''
  parser = argparse.ArgumentParser(
      description='Instance segmentation by affinity derivation and graph merge')
  parser.add_argument('--sem_ckpt_path', type=str, default='',
                      help="Path to the saved semantic checkpoint.")
  parser.add_argument('--aff_ckpt_path', type=str, default='',
                      help="Path to the saved affinity checkpoint.")
  parser.add_argument('--demo_data_dir', type=str, default='',
                      help="directory contains test pictures.")
  parser.add_argument('--out_dir', type=str, default='',
                      help="the output result directory.")
  parser.add_argument('--tmp_dir', type=str, default='',
                      help="the directory for temporary results. if not specified, use out_dir")
  parser.add_argument('--evaluation_result_out_dir', type=str,
                      default='', help="the output result for evaluation directory.")
  parser.add_argument('--class_name_list', type=str,
                      default='cityscapes_9cls', help="the list of class names.")
  parser.add_argument('--post_processing_exec', type=str,
                      default='', help="post processing exec.")

  parser.add_argument('--num_classes', type=int, default=9,
                      help="number of classes, including background.")
  parser.add_argument('--num_split_masks', type=int,
                      default=3, help="number of split masks.")
  parser.add_argument('--pad_to_size', type=int, default=513,
                      help="pad the input image to.")
  parser.add_argument('--semantic_stride', type=int, default=8,
                      help="output stride, relative to the input image.")
  parser.add_argument('--affinity_stride', type=int, default=8,
                      help="output stride, relative to the input image.")
  parser.add_argument('--merge_before_post_process', type=int, default=2,
                      choices=[1, 2, 4], help="1/2/4 times force local merge before post process.")
  parser.add_argument('--inference_process_num', type=int,
                      default=1, help="inference process num.")
  parser.add_argument('--post_processing_process_num',
                      type=int, default=2, help="post processing process num.")
  parser.add_argument('--scale_to_size', type=int, default=513,
                      help="scale sub region to, suggested 512 or 513.")
  parser.add_argument('--merge_region_size', type=int, default=32,
                      help="size to find connected area.")

  parser.add_argument('--fresh_run', action="store_true", help="ignore all existing results")
  parser.add_argument('--show_mask_on_image',
                      action="store_true", help="show mask on image or not")
  parser.add_argument('--use_sem_flip', action="store_true",
                      help="use flipped image as inputs")
  parser.add_argument('--use_aff_flip', action="store_true",
                      help="use flipped image as inputs")
  parser.add_argument('--auto_upsample', action="store_true",
                      help="auto_upsample for different image parts")
  parser.add_argument('--bike_prob_refine', action="store_true",
                      help="refine the probabilities of bicycle and motorbike classes")

  args = parser.parse_args()
  return args


def get_split_mask_label(num_of_split):
  '''
  split into groups according to semantic label
  '''
  if num_of_split == 1:
    return np.array([0, 1, 1, 1, 1, 1, 1, 1, 1])
  elif num_of_split == 2:
    return np.array([0, 1, 1, 2, 2, 2, 2, 1, 1])
  elif num_of_split == 3:
    return np.array([0, 1, 1, 2, 2, 2, 2, 3, 3])

  return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])


def preprocess_image(image):
  '''
  the input image is hxwx3 numpy array
  the output image is 1xhxwx3 float numpy array
  '''
  channel_means = [123.68, 116.779, 103.939]
  image_modified = image.astype(np.float)
  image_modified = image_modified - channel_means
  return image_modified


def postprocess_prob(prob):
  '''
  input is 4d prob
  output is 3d segmentation index
  '''
  max_indicator = np.argmax(prob, -1)
  return max_indicator.astype(np.uint8)


def get_overlapped_region(offset_x, offset_y, img_height, img_width):
  '''
  find the overlapped region
  '''
  x1start = max(0, offset_x)
  x1end = min(img_height, img_height + offset_x)
  y1start = max(0, offset_y)
  y1end = min(img_width, img_width + offset_y)
  x2start = max(0, -offset_x)
  x2end = min(img_height, img_height - offset_x)
  y2start = max(0, -offset_y)
  y2end = min(img_width, img_width - offset_y)

  return x1start, x1end, y1start, y1end, x2start, x2end, y2start, y2end


def average_affinity_prob_array(arr, strides):
  '''
  average affinity probability array
  '''
  img_height = arr.shape[0]
  img_width = arr.shape[1]
  offset_y = [-1, 0, 1, -1, 1, -1, 0, 1]
  offset_x = [-1, -1, -1, 0, 0, 1, 1, 1]
  for i, stride in enumerate(strides):
    for j in range(len(offset_x) // 2):  # num_neighbor / 2
      tmp1 = arr[:, :, i, 7 - j]
      tmp2 = arr[:, :, i, j]
      x1start, x1end, y1start, y1end, x2start, x2end, y2start, y2end = \
        get_overlapped_region(
            stride * offset_x[j], stride * offset_y[j], img_height, img_width)

      tmp = (tmp1[x1start:x1end, y1start:y1end] +
             tmp2[x2start:x2end, y2start:y2end]) / 2
      arr[x1start:x1end, y1start:y1end, i, 7 - j] = tmp
      arr[x2start:x2end, y2start:y2end, i, j] = tmp
  return arr


def class_probability_array(arr, strides):
  '''
  use semantic probability to refine the affinity probability
  '''
  img_height = arr.shape[0]
  img_width = arr.shape[1]
  cat_idx = get_split_mask_label(3) # regardless of num_of_split, and force to 3
  cats = 4
  new_arr = np.zeros((img_height, img_width, cats), dtype=np.float32)
  for j in range(cats):
    new_arr[:, :, j] = np.sum(arr[:, :, cat_idx == j], axis=2)
  arr = new_arr
  tmp_sem_mask = np.argmax(arr, axis=2)

  offset_y = [-1, 0, 1, -1, 1, -1, 0, 1]
  offset_x = [-1, -1, -1, 0, 0, 1, 1, 1]
  res = np.zeros((img_height, img_width, len(
      strides), len(offset_x)), dtype=np.float32)
  for i, stride in enumerate(strides):
    for j in range(len(offset_x) // 2):  # num_neighbor / 2
      x1start, x1end, y1start, y1end, x2start, x2end, y2start, y2end = \
        get_overlapped_region(
            stride * offset_x[j], stride * offset_y[j], img_height, img_width)

      tmp = arr[x1start:x1end, y1start:y1end] * \
            arr[x2start:x2end, y2start:y2end]
      tmp = tmp[:, :, 1:].sum(axis=2)
      is_same_class_group = tmp_sem_mask[x1start:x1end, y1start:y1end] == \
                            tmp_sem_mask[x2start:x2end, y2start:y2end]

      tmp = tmp * is_same_class_group
      res[x1start:x1end, y1start:y1end, i, 7 - j] = tmp
      res[x2start:x2end, y2start:y2end, i, j] = tmp
  return res


def generate_split_mask_list(mask_group, ori_prob=None):
  '''
  generate split mask list
  '''
  split_num = np.max(mask_group)
  mask_list = []

  cats = np.max(mask_group).astype('int32')+1
  new_arr = np.zeros((ori_prob.shape[0], ori_prob.shape[1], cats), dtype=np.float32)
  for j in range(cats):
    new_arr[:, :, j] = np.sum(ori_prob[:, :, mask_group == j], axis=2)
  tmp_sem_mask = np.argmax(new_arr, axis=2)
  for i in range(split_num):
    tmp_mask = np.copy(tmp_sem_mask)
    tmp = np.arange(cats)[np.arange(cats) != (i + 1)]
    tmp_mask[np.isin(tmp_mask, tmp)] = 0
    mask_list.append(tmp_mask)

  return mask_list


def merge_before_post_process(args, tmp_strides, tmp_aff_prob, tmp_mask_prob):
  '''
  force local merge before post process
  '''
  tmp_aff_prob.resize((tmp_aff_prob.shape[0], tmp_aff_prob.shape[1], len(tmp_strides), 8))

  if args.merge_before_post_process == 2:  # perform 2 by 2 merge
    tmp_strides = tmp_strides[1:] // 2
    tmp_aff_prob = merge_2by2(tmp_aff_prob).astype(np.float32)
    tmp_mask_prob = merge_semantic_2by2(tmp_mask_prob)

  elif args.merge_before_post_process == 4:  # perform 4 by 4 merge
    tmp_strides = tmp_strides[2:] // 4
    tmp_aff_prob = merge_4by4(tmp_aff_prob).astype(np.float32)
    tmp_mask_prob = merge_semantic_4by4(tmp_mask_prob)

  return tmp_strides, tmp_aff_prob, tmp_mask_prob


def process_probability_array(tmp_aff_prob, tmp_mask_prob, tmp_strides):
  '''
  process affinity probability
  '''
  tmp_prob = average_affinity_prob_array(tmp_aff_prob, tmp_strides)
  tmp_class_prob = class_probability_array(tmp_mask_prob, tmp_strides)

  return tmp_prob, tmp_class_prob


def network_inference(args, process_id, inf_request_queue, inf_response_queues):
  '''
  network inference process, must have GPU
  '''
  os.environ['CUDA_VISIBLE_DEVICES'] = str(process_id)
  import tensorflow as tf
  import semantic_seg
  import affinity_seg

  sem_graph = tf.Graph()
  with sem_graph.as_default():
    sem_image_tensor = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3], name='image_tensor')
    sem_prob_tensor = semantic_seg.prob_for_one_image(sem_image_tensor, args.num_classes,
                                                      pad_to_size=args.pad_to_size,
                                                      output_stride=args.semantic_stride)
    sess1 = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess1, args.sem_ckpt_path)

  aff_graph = tf.Graph()
  with aff_graph.as_default():
    aff_image_tensor = tf.placeholder(
        dtype=tf.float32, shape=[None, None, None, 3], name='image_tensor')
    aff_prob_tensor = affinity_seg.prob_for_one_image(aff_image_tensor,
                                                      pad_to_size=args.pad_to_size,
                                                      output_stride=args.affinity_stride)
    sess2 = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess2, args.aff_ckpt_path)

  while True:
    inf_request = inf_request_queue.get()

    if inf_request['type'] == 'semantic':
      res = sess1.run(sem_prob_tensor,
                      feed_dict={sem_image_tensor: inf_request['input']})
    elif inf_request['type'] == 'affinity':
      res = sess2.run(aff_prob_tensor,
                      feed_dict={aff_image_tensor: inf_request['input']})
    elif inf_request['type'] == 'terminate':
      break

    if res.nbytes > 1 * 1024 * 1024 * 1024:
      res_filename = args.tmp_dir + '/' + str(process_id) + '_' + \
                     str(inf_request['process_id']) + '.npy'
      np.save(res_filename, res)
      inf_response = {
          'type': inf_request['type'],
          'output': res_filename
      }
    else:
      inf_response = {
          'type': inf_request['type'],
          'output': res
      }
    inf_response_queues[inf_request['process_id']].put(inf_response)

  sess1.close()
  sess2.close()


def parse_inf_output(output):
  '''
  if the output is from file, read it into memory
  '''
  if isinstance(output, str):
    res_filename = output
    output = np.load(res_filename)
    os.remove(res_filename)
  return output

MARGIN_FOR_PATCH = 256
MARGIN_EFFECTIVE = 128

def split_inf_input(image_patch):
  '''
  split the input image if it is too big
  '''
  w_max = 1024 * 1024 * 2 / image_patch.shape[0] / image_patch.shape[1]
  num = np.ceil((image_patch.shape[2] - MARGIN_FOR_PATCH) /
                (w_max - MARGIN_FOR_PATCH)).astype('int32')
  width = np.ceil((image_patch.shape[2] - 256) / num + 256).astype('int32')
  splited_list = []
  x_start = 0
  for j in range(num):
    if j < num - 1:
      splited_list.append(image_patch[:, :, x_start:x_start+width, :])
      x_start = x_start + width - MARGIN_FOR_PATCH
    else:
      splited_list.append(image_patch[:, :, x_start:, :])
  return splited_list


def merge_inf_output(mask_list):
  '''
  merge the inference output if the input has been split
  '''
  num = len(mask_list)
  mask_batch = []
  for j in range(num):
    if j == 0:
      mask_batch = mask_list[j][:, :, :-MARGIN_EFFECTIVE, :]
    elif j < num - 1:
      mask_batch = np.concatenate((mask_batch,
                                   mask_list[j][:, :, MARGIN_EFFECTIVE:-MARGIN_EFFECTIVE, :]),
                                  axis=2)
    else:
      mask_batch = np.concatenate((mask_batch, mask_list[j][:, :, MARGIN_EFFECTIVE:, :]), axis=2)
  return mask_batch


def get_inference_result(process_id, input_batch, inf_request_queue,
                         inf_response_queue, request_type):
  '''
  get inference result, put the input in the queue and get the output from another queue
  '''
  inf_request = {
      'type': request_type,
      'process_id': process_id,
      'input': input_batch
  }
  inf_request_queue.put(inf_request)
  inf_response = inf_response_queue.get()
  return parse_inf_output(inf_response['output'])


def inference_by_request(process_id, image_batch, inf_request_queue,
                         inf_response_queue, request_type):
  '''
  inference
  '''
  assert request_type == 'semantic' or request_type == 'affinity', \
         'request type should be semantic or affinity'

  if image_batch.size > 4096 * 2048 * 3:
    if image_batch.shape[0] == 2:
      img_list = [image_batch[[0], :, :, :], image_batch[[1], :, :, :]]
    else:
      img_list = [image_batch]

    result_list = []
    for img in img_list:
      if img.size > 4096 * 2048 * 3:
        splited_list = split_inf_input(img)
        mask_list = []
        for splited in splited_list:
          mask_batch_splited = get_inference_result(process_id, splited, inf_request_queue,
                                                    inf_response_queue, request_type)
          mask_list.append(mask_batch_splited)
        img_mask_batch = merge_inf_output(mask_list)
      else:
        img_mask_batch = get_inference_result(process_id, img, inf_request_queue,
                                              inf_response_queue, request_type)
      result_list.append(img_mask_batch)

    if image_batch.shape[0] == 2:
      mask_batch = np.concatenate((result_list[0], result_list[1]), axis=0)
    else:
      mask_batch = result_list[0]

  else:
    mask_batch = get_inference_result(process_id, image_batch, inf_request_queue,
                                      inf_response_queue, request_type)

  return mask_batch


def get_semantic_wait(process_id, input_image, inf_request_queue, inf_response_queue, use_flip):
  '''
  get semantic segmentation output
  '''
  if use_flip:
    flipped_image = np.flip(input_image, axis=-2)
    image_batch = np.stack((input_image, flipped_image), axis=0)
  else:
    image_batch = np.expand_dims(input_image, 0)

  mask_batch = inference_by_request(process_id, image_batch, inf_request_queue,
                                    inf_response_queue, 'semantic')

  if use_flip:
    mask_prob_original = mask_batch[0]
    mask_prob_flipped = np.flip(mask_batch[1], axis=-2)
    mask_prob = np.stack((mask_prob_original, mask_prob_flipped), axis=0)
    mask_prob = np.mean(mask_prob, axis=0)
  else:
    mask_prob = np.squeeze(mask_batch, 0)

  return mask_prob


def get_affinity_wait(process_id, input_image, inf_request_queue, inf_response_queue,
                      use_flip, inverse_id):
  '''
  get pixel affinity
  '''
  if use_flip:
    flipped_image = np.flip(input_image, axis=-2)
    image_batch = np.stack((input_image, flipped_image), axis=0)
  else:
    image_batch = np.expand_dims(input_image, 0)

  aff_prob_batch = inference_by_request(process_id, image_batch, inf_request_queue,
                                        inf_response_queue, 'affinity')

  if use_flip:
    aff_prob_original = aff_prob_batch[0]
    aff_prob_flipped = aff_prob_batch[1]
    aff_prob_flipped = np.flip(aff_prob_flipped[:, :, inverse_id], -2)
    aff_prob = np.stack((aff_prob_original, aff_prob_flipped), axis=0)
    aff_prob = np.mean(aff_prob, axis=0)
  else:
    aff_prob = np.squeeze(aff_prob_batch, axis=0)

  return aff_prob

def get_new_axis(xmin, xmax, zoomfac, x_b, tgt_x):
  '''
  get the new axis to make sure the size is no smaller than tgt_x
  '''
  minx = tgt_x/zoomfac
  cur_width = xmax-xmin+1
  if cur_width >= minx:
    return xmin, xmax
  else:
    margin = minx-cur_width
    tmp_xmin = xmin-margin/2
    tmp_xmax = xmax+margin/2
    if tmp_xmax > x_b:
      tmp_xmin = x_b - minx
      tmp_xmax = x_b-1
    if tmp_xmin < 0:
      tmp_xmax = minx-1
      tmp_xmin = 0
    return int(tmp_xmin), int(tmp_xmax)

def post_processing(args, process_id, counter, demo_files_queue,
                    inf_request_queue, inf_response_queue):
  '''
  the post processing process
  '''
  warnings.filterwarnings('ignore', '.*output shape of zoom.*')
  mask_palette = get_default_palette.get_default_palette()

  if args.use_aff_flip:
    inverse_index = np.array([2, 1, 0, 4, 3, 7, 6, 5]).reshape((8))
    inverse_id = np.array([2, 1, 0, 4, 3, 7, 6, 5]).reshape((8))
    for k in range(1, 7):
      inverse_id = np.concatenate((inverse_id, inverse_index + k * 8), axis=0)
  else:
    inverse_id = None

  get_demp_file_attemps = 0
  while True:
    # get demo_file from multiprocessing.Queue
    try:
      demo_file = demo_files_queue.get_nowait()
    except:
      get_demp_file_attemps += 1
      if get_demp_file_attemps < 3:
        time.sleep(random.random())
        continue
      else:
        break
    else:
      get_demp_file_attemps = 0

    _t_start = time.time()

    output_file_name = demo_file.replace(
        args.demo_data_dir, args.out_dir).replace('.jpg', '.png')
    tmp_output_file_name = output_file_name.replace(args.out_dir, args.tmp_dir)

    img = Image.open(demo_file)
    image = np.array(img)

    processed_image = preprocess_image(image)
    mask_prob = get_semantic_wait(
        process_id, processed_image, inf_request_queue, inf_response_queue, args.use_sem_flip)

    mask = postprocess_prob(mask_prob)

    mask_image = Image.fromarray(mask, mode='P')
    mask_image.putpalette(mask_palette)
    mask_image.save(tmp_output_file_name)

    ori_mask = np.copy(mask)

    mask_label = get_split_mask_label(args.num_split_masks)
    mask_list = generate_split_mask_list(mask_label, mask_prob)

    inst_count = 0
    for mask_i, mask in enumerate(mask_list):
      tmp_max_inst_num = 0
      inst_result_image = np.zeros(
          (ori_mask.shape[0], ori_mask.shape[1]), dtype=np.uint8)
      confidence_all = np.empty(0)
      confidence_all = np.append(
          confidence_all, np.array([0], dtype='float32'))

      connect_areas = find_connect_area(
          mask, args.merge_region_size, expend_border=True).astype('uint8')
      connective_area_num = np.max(connect_areas)

      for j in range(1, connective_area_num + 1):
        connective_area_mask = connect_areas == j

        idxs = np.where(connective_area_mask)
        xmin = np.min(idxs[0])
        xmax = np.max(idxs[0])
        ymin = np.min(idxs[1])
        ymax = np.max(idxs[1])

        area_height = xmax - xmin + 1
        area_width = ymax - ymin + 1
        zoom_factor_max = 4.
        zoom_factor_min = np.maximum(128. / area_height, 128. / area_width)
        zoom_factor = np.minimum(
            zoom_factor_max, args.scale_to_size / float(area_height))
        zoom_factor = np.maximum(zoom_factor, zoom_factor_min)
        zoom_factor = np.maximum(zoom_factor, 1)

        if not args.auto_upsample:
          zoom_factor = 1.
          xmin, xmax = get_new_axis(
              xmin, xmax, zoom_factor, mask.shape[0], 65)
          ymin, ymax = get_new_axis(
              ymin, ymax, zoom_factor, mask.shape[1], 65)

        cropped_image = processed_image[xmin:xmax + 1, ymin:ymax + 1, :]
        zoomed_image = zoom(cropped_image.astype(
            'float32'), (zoom_factor, zoom_factor, 1), order=1)
        tmp_aff_prob = get_affinity_wait(process_id, zoomed_image, inf_request_queue,
                                         inf_response_queue, args.use_aff_flip, inverse_id)
        tmp_mask_prob = get_semantic_wait(process_id, zoomed_image, inf_request_queue,
                                          inf_response_queue, args.use_sem_flip)
        tmp_strides = np.array([1, 2, 4, 8, 16, 32, 64], dtype=np.int32)

        tmp_strides, tmp_aff_prob, tmp_mask_prob = merge_before_post_process(args, tmp_strides,
                                                                             tmp_aff_prob,
                                                                             tmp_mask_prob)

        tmp_mask = postprocess_prob(tmp_mask_prob)
        tmp_prob, tmp_class_prob = process_probability_array(tmp_aff_prob, tmp_mask_prob,
                                                             tmp_strides)

        inst_input_bin_file = tmp_output_file_name.replace(
            '.png', '%d_%0.2d.input.bin' % (mask_i, j))

        with open(inst_input_bin_file, 'wb') as file:
          shape = np.array(tmp_prob.shape, dtype=np.int32)
          file.write(shape.tobytes())
          file.write(tmp_strides.tobytes())
          file.write(tmp_prob.tobytes())
          file.write(tmp_class_prob.tobytes())
          file.write(tmp_mask.tobytes())

        # perform post processing via graph merge
        inst_out_bin_file = inst_input_bin_file.replace(
            '.input.bin', '.output.bin')
        tmp_inst_out_conf_file = inst_input_bin_file.replace(
            '.input.bin', '.inst.confidence')
        tmp_inst_out_conf_txt_file = inst_input_bin_file.replace(
            '.input.bin', '.inst.confidence.txt')
        bike_prob_refine_flag = 'BIKE_PROB_REFINE_TRUE'
        if not args.bike_prob_refine:
          bike_prob_refine_flag = 'BIKE_PROB_REFINE_FALSE'
        command = '"' + args.post_processing_exec + '" ' \
                  + inst_input_bin_file + ' ' \
                  + inst_out_bin_file + ' ' \
                  + tmp_inst_out_conf_file + ' ' \
                  + tmp_inst_out_conf_txt_file + ' ' \
                  + bike_prob_refine_flag
        os.system(command)

        tmp_inst_result_image = np.fromfile(
            inst_out_bin_file, dtype=np.uint8)

        if args.merge_before_post_process > 0:  # instance mask upsample
          t_merge_time = args.merge_before_post_process
          tmp_inst_result_image.resize(
              zoomed_image.shape[0] // t_merge_time, zoomed_image.shape[1] // t_merge_time)
          upsample_res = np.zeros(
              (zoomed_image.shape[0], zoomed_image.shape[1]), dtype=np.uint8)
          tmp_inst_result_image = np.repeat(
              tmp_inst_result_image, t_merge_time, axis=0)
          tmp_inst_result_image = np.repeat(
              tmp_inst_result_image, t_merge_time, axis=1)
          upsample_res[:tmp_inst_result_image.shape[0],
                       :tmp_inst_result_image.shape[1]] = tmp_inst_result_image[:, :]
          tmp_inst_result_image = upsample_res
        else:
          tmp_inst_result_image.resize(
              zoomed_image.shape[0], zoomed_image.shape[1])

        # result tmp_prob downsample and combine
        cur_inst_num = np.max(tmp_inst_result_image)
        tmp_inst_result_image = (tmp_inst_result_image > 0).astype(
            'int32') * tmp_max_inst_num + tmp_inst_result_image
        tmp_max_inst_num += cur_inst_num
        cropped_inst_result_image = zoom(
            tmp_inst_result_image, (1. / zoom_factor, 1. / zoom_factor), order=0)

        tmp_whole_image = np.zeros(
            (connective_area_mask.shape[0], connective_area_mask.shape[1]), dtype=np.uint8)
        tmp_whole_image[xmin:xmax + 1, ymin:ymax +
                        1] = cropped_inst_result_image
        tmp_whole_image[np.logical_not(connective_area_mask)] = 0
        inst_result_image = inst_result_image + tmp_whole_image
        tmp_confidence_arr = np.fromfile(
            tmp_inst_out_conf_file, dtype='float32')
        confidence_all = np.append(confidence_all, tmp_confidence_arr[1:])

      result_mask_path = './instance_masks/'
      result_txt_path = './result_txt/'
      save_file_name = os.path.basename(demo_file)[:-4]

      mask_label = get_split_mask_label(args.num_split_masks)
      inst_result_image = \
          gen_evaluation_file.gen_evaluation_file(ori_mask, inst_result_image,
                                                  confidence_all, result_mask_path,
                                                  result_txt_path, save_file_name,
                                                  additive=True if mask_i > 0 else False,
                                                  slide_semantic=True,
                                                  mask_label=mask_label,
                                                  mask_idx=mask_i,
                                                  result_path=args.evaluation_result_out_dir)

      inst_result_image[inst_result_image >
                        0] = inst_result_image[inst_result_image > 0] + inst_count
      inst_count = inst_count + np.max(inst_result_image)
      inst_result_image = Image.fromarray(inst_result_image, mode='P')
      inst_result_image.putpalette(mask_palette)
      inst_result_image.save(tmp_output_file_name.replace(
          '.png', '.inst' + str(mask_i) + '.png'))

      for j in range(1, connective_area_num + 1):
        os.remove(tmp_output_file_name.replace(
            '.png', '%d_%0.2d.input.bin' % (mask_i, j)))
        os.remove(tmp_output_file_name.replace(
            '.png', '%d_%0.2d.output.bin' % (mask_i, j)))
        os.remove(tmp_output_file_name.replace(
            '.png', '%d_%0.2d.inst.confidence' % (mask_i, j)))
        os.remove(tmp_output_file_name.replace(
            '.png', '%d_%0.2d.inst.confidence.txt' % (mask_i, j)))

    if args.show_mask_on_image:
      combine_file_name = output_file_name.replace('.png', '_combine.jpg')
      combined_image = show_mask.combine(
          demo_file, tmp_output_file_name, args.class_name_list)
      combined_image.save(combine_file_name)

    # combile instance mask with image
    combine_file_name = output_file_name.replace('.png', '_combine.inst.jpg')
    combined_image = show_mask.combine(demo_file, tmp_output_file_name.replace(
        '.png', '.inst' + str(0) + '.png'), args.class_name_list, has_legend=False)
    combined_image.save(combine_file_name)

    for num in range(1, len(mask_list)):
      combined_image = show_mask.combine(combine_file_name, tmp_output_file_name.replace(
          '.png', '.inst' + str(num) + '.png'), args.class_name_list, has_legend=False)
      combined_image.save(combine_file_name)

    for idx in range(len(mask_list)):
      os.remove(tmp_output_file_name.replace('.png', '.inst' + str(idx) + '.png'))
    os.remove(tmp_output_file_name)

    _t_end = time.time()
    with counter.get_lock():
      counter.value += 1
      print('post processing {} finished image {} ({}) in {:.2f}s'.format(
          process_id, counter.value, demo_file, _t_end - _t_start))


def start_network_inf_processes(args, process_num, inf_request_queue, inf_response_queues):
  '''
  start network inference process. GPU required
  '''
  processes = []
  for i in range(process_num):
    processes.append(multiprocessing.Process(target=network_inference,
                                             args=[args, i,
                                                   inf_request_queue,
                                                   inf_response_queues]))
  for process in processes:
    process.start()

  return processes


def start_post_processing_processes(args, process_num, inf_request_queue, inf_response_queues):
  '''
  start post processing process
  '''
  total_exist_file = 0
  total_pending_file = 0
  demo_files_queue = multiprocessing.Queue()
  for root, _, files in os.walk(args.demo_data_dir):
    dst_folder = root.replace(args.demo_data_dir, args.out_dir)
    if not os.path.exists(dst_folder):
      os.makedirs(dst_folder)
      print('created folder ' + dst_folder)
    dst_folder = root.replace(args.demo_data_dir, args.tmp_dir)
    if not os.path.exists(dst_folder):
      os.makedirs(dst_folder)
      print('created folder ' + dst_folder)

    for filename in files:
      if filename.endswith('.png') or filename.endswith('.jpg'):
        if args.fresh_run:
          total_pending_file += 1
          demo_files_queue.put(os.path.join(root, filename))
        else:
          # check whether there exists results
          full_name = os.path.join(root, filename)
          inst_comb_name = full_name.replace(args.demo_data_dir, args.out_dir)
          inst_comb_name = inst_comb_name.replace('.jpg', '.png')
          inst_comb_name = inst_comb_name.replace('.png', '_combine.inst.jpg')
          if os.path.exists(inst_comb_name):
            total_exist_file += 1
          else:
            total_pending_file += 1
            demo_files_queue.put(os.path.join(root, filename))

  print("exist: " + str(total_exist_file))
  print("to be processed: " + str(total_pending_file))

  counter = multiprocessing.Value('i', 0)
  processes = []
  for i in range(process_num):
    processes.append(multiprocessing.Process(target=post_processing,
                                             args=[args, i, counter,
                                                   demo_files_queue,
                                                   inf_request_queue,
                                                   inf_response_queues[i]]))
  for process in processes:
    process.start()

  return processes


def join_all_processes(processes):
  '''
  join all processes
  '''
  for process in processes:
    process.join()


def check_dir_and_create(directory):
  '''
  if the directory does not exist, create it
  '''
  if not os.path.exists(directory):
    os.makedirs(directory)
    print('created folder ' + directory)


def main():
  '''
  main function
  '''
  args = parse_args()
  print('Called with args:')
  print(args)

  assert args.sem_ckpt_path, '`sem_ckpt_path` is missing.'
  assert glob.glob(args.sem_ckpt_path + '*'), 'cannot find ' + \
      args.sem_ckpt_path
  assert args.demo_data_dir, '`demo_data_dir` is missing'
  assert args.out_dir, '`out_dir` is missing'
  assert args.evaluation_result_out_dir, 'evaluation_result_out_dir is missing'
  assert args.pad_to_size == 0 or args.pad_to_size > 512, 'pad_to_size should > 512'
  assert args.semantic_stride == 16 or args.semantic_stride == 8, 'semantic_stride must be 16 or 8'
  assert args.affinity_stride == 16 or args.affinity_stride == 8, 'affinity_stride must be 16 or 8'
  assert os.path.exists(
      args.post_processing_exec), 'cannot find post processing exec at ' + args.post_processing_exec

  if args.tmp_dir == '':
    args.tmp_dir = args.out_dir

  check_dir_and_create(args.out_dir)
  check_dir_and_create(args.tmp_dir)
  check_dir_and_create(args.evaluation_result_out_dir)
  check_dir_and_create(args.evaluation_result_out_dir + '/instance_masks')
  check_dir_and_create(args.evaluation_result_out_dir + '/result_txt')

  global_start_time = time.time()

  inf_request_queue = multiprocessing.Queue()
  inf_response_queues = []
  for _ in range(args.post_processing_process_num):
    inf_response_queues.append(multiprocessing.Queue())

  network_inf_processes = []
  network_inf_processes += start_network_inf_processes(
      args, args.inference_process_num, inf_request_queue, inf_response_queues)

  post_processes = []
  post_processes += start_post_processing_processes(
      args, args.post_processing_process_num, inf_request_queue, inf_response_queues)

  join_all_processes(post_processes)
  for _ in range(args.inference_process_num):
    inf_request_queue.put({
        'type': 'terminate'
    })
  join_all_processes(network_inf_processes)

  global_end_time = time.time()
  print('total run time: {:.2f}s'.format(global_end_time - global_start_time))


if __name__ == '__main__':
  main()
