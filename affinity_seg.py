'''
This file implements semantic segmentation, based on DeepLab v3
'''
import tensorflow as tf
import resnet_v1_base

def affinity_seg(inputs, output_stride=16):
  '''
  inputs for training should be in the shape of [batch_size, height, width, 3]
  output_stride should be either 16 or 8
  '''
  assert output_stride == 16 or output_stride == 8, "output_stride should be 16 or 8"
  with tf.variable_scope('resnet_v1_101'):
    net = resnet_v1_base.resnet_head(inputs)
    net = resnet_v1_base.resnet_block(net, 64, 256, 2, 1, 3, scope='block1')
    if output_stride == 16:
      net = resnet_v1_base.resnet_block(net, 256, 512, 2, 1, 4, scope='block2')
      net = resnet_v1_base.resnet_block(net, 512, 1024, 1, 1, 23, scope='block3')
      net = resnet_v1_base.resnet_block(net, 1024, 2048, 1, 2, 3, scope='block4',
                                        rate_multiple=[1, 2, 4])
    else:
      net = resnet_v1_base.resnet_block(net, 256, 512, 1, 1, 4, scope='block2')
      net = resnet_v1_base.resnet_block(net, 512, 1024, 1, 2, 23, scope='block3')
      net = resnet_v1_base.resnet_block(net, 1024, 2048, 1, 4, 3, scope='block4',
                                        rate_multiple=[1, 2, 4])

  with tf.variable_scope('deeplab_v3'):
    affinity_out = resnet_v1_base.aspp_layer(net, output_stride, scope="aspp_inst")
    score_affinity = resnet_v1_base.resnet_score_layer(affinity_out, 7*8, scope='logits_inst')

  return score_affinity

def predict(input_dict, output_stride=16):
  '''
  predict affinity info for one batch
  '''
  image = input_dict['image']
  org_shape = tf.shape(image)

  affinity = affinity_seg(image, output_stride)
  curr_shape = tf.shape(affinity)
  affinity = tf.image.resize_bilinear(affinity, [curr_shape[1] * output_stride,
                                                 curr_shape[2] * output_stride])

  affinity = tf.slice(affinity, [0, 0, 0, 0], [org_shape[0], org_shape[1], org_shape[2], 7*8])

  return affinity

def _pad_image(image, min_height, min_width):
  '''
  image is a 4-D tensor
  (zero) pad the image if height < min_height or width < min_width
  '''
  org_height = tf.shape(image)[1]
  org_width = tf.shape(image)[2]
  padded_row = tf.maximum(0, min_height - org_height)
  padded_col = tf.maximum(0, min_width - org_width)

  image = tf.pad(image, [[0, 0], [0, padded_row], [0, padded_col], [0, 0]])
  image.set_shape((None, None, None, 3))

  return image

def prob_for_one_image(image, pad_to_size=0, output_stride=16):
  '''
  get affinity probability of input image
  '''
  if pad_to_size > 0:
    assert pad_to_size > 512, "pad_to_size should > 512"
    org_shape = tf.shape(image)
    image = _pad_image(image, pad_to_size, pad_to_size)

  input_dict = {}
  input_dict['image'] = image
  score = predict(input_dict, output_stride)
  if pad_to_size > 0:
    score = tf.slice(score, [0, 0, 0, 0], [org_shape[0], org_shape[1], org_shape[2], 7*8])

  prob = tf.sigmoid(score)

  return prob
