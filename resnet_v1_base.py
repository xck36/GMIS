'''
This file implements the bacics of resnet and some basic components of Deeplab v3.
Most code are duplicated from contrib.slim, but as we need to modify the code,
they are provided in this file again
We use the tensorflow slim implementation of resent, the main difference is the
stride is 2 (is necessary) at the last unit (conv2) of a block, instead of using the
stride of 2 at the first unit of a block, which may be more efficient according to
https://github.com/tensorflow/tensorflow/issues/9387
And the pretrained model comes from tensorflow model slim.
'''
import tensorflow as tf

def get_resnet_batch_norm_params():
  '''
  get common parameters for batch normalization
  '''
  resnet_batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 1e-5,
      'scale': True,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': False,
  }
  return resnet_batch_norm_params

def get_resnet_conv2d_params(include_relu=True, padding='SAME'):
  '''
  get common parameters for conv2d layer
  '''
  resnet_conv2d_params = {
      'padding': padding,
      'activation_fn': tf.nn.relu if include_relu else None,
      # 'weights_initializer': tf.variance_scaling_initializer(),
      'weights_regularizer': tf.contrib.layers.l2_regularizer(0.0001),
      'biases_initializer': None,
      'normalizer_fn': tf.contrib.layers.batch_norm,
      'normalizer_params': get_resnet_batch_norm_params(),
  }
  return resnet_conv2d_params

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """
  see more details from
  https://github.com/tensorflow/models/blob/master/slim/nets/resnet_utils.py
  """
  assert stride > 1, 'please use conv2d for stride=1'

  kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
  pad_total = kernel_size_effective - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  output = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                                    rate=rate, scope=scope,
                                    **get_resnet_conv2d_params(True, padding='VALID'))
  return output

def resnet_head(inputs):
  '''
  definition for the head of resnet, including the first conv layer and max pool
  '''
  conv1 = conv2d_same(inputs, 64, 7, 2, scope='conv1')
  pool1 = tf.contrib.layers.max_pool2d(conv1, [3, 3], 2, padding='SAME', scope='pool1')
  return pool1

def resnet_block_first_unit(inputs, channel_in, channel_out, rate, scope='bottleneck_v1'):
  '''
  the first unit of a residual block,
  upsampling is required for the input before adding to the residue
  '''
  with tf.variable_scope(scope):
    channel_middle = channel_out // 4
    if channel_in == channel_out:
      shortcut = inputs
    else:
      shortcut = tf.contrib.layers.conv2d(inputs, channel_out, [1, 1], stride=1,
                                          rate=rate, scope='shortcut',
                                          **get_resnet_conv2d_params(False))

    residual = tf.contrib.layers.conv2d(inputs, channel_middle, [1, 1], stride=1,
                                        rate=rate, scope='conv1',
                                        **get_resnet_conv2d_params())
    residual = tf.contrib.layers.conv2d(residual, channel_middle, [3, 3], stride=1,
                                        rate=rate, scope='conv2',
                                        **get_resnet_conv2d_params())
    residual = tf.contrib.layers.conv2d(residual, channel_out, [1, 1], stride=1,
                                        rate=rate, scope='conv3',
                                        **get_resnet_conv2d_params(False))

    output = tf.nn.relu(shortcut + residual)

    return output

def resnet_block_middle_unit(inputs, channel_out, rate, scope='bottleneck_v1'):
  '''
  the middle unit of a residual block,
  upsampling is required for the input before adding to the residue
  '''
  with tf.variable_scope(scope):
    channel_middle = channel_out // 4
    shortcut = inputs

    residual = tf.contrib.layers.conv2d(inputs, channel_middle, [1, 1], stride=1,
                                        rate=rate, scope='conv1',
                                        **get_resnet_conv2d_params())
    residual = tf.contrib.layers.conv2d(residual, channel_middle, [3, 3], stride=1,
                                        rate=rate, scope='conv2',
                                        **get_resnet_conv2d_params())
    residual = tf.contrib.layers.conv2d(residual, channel_out, [1, 1], stride=1,
                                        rate=rate, scope='conv3',
                                        **get_resnet_conv2d_params(False))

    output = tf.nn.relu(shortcut + residual)

    return output

def resnet_block_last_unit(inputs, channel_out, stride, rate, scope='bottleneck_v1'):
  '''
  the last unit of a residual block, the stride may not be 1
  '''
  with tf.variable_scope(scope):
    channel_middle = channel_out // 4
    if stride == 1:
      shortcut = inputs
    else:
      shortcut = tf.contrib.layers.max_pool2d(inputs, [1, 1], stride, padding='SAME')

    residual = tf.contrib.layers.conv2d(inputs, channel_middle, [1, 1], stride=1,
                                        rate=rate, scope='conv1',
                                        **get_resnet_conv2d_params())
    if stride == 1:
      residual = tf.contrib.layers.conv2d(residual, channel_middle, [3, 3], stride=1,
                                          rate=rate, scope='conv2',
                                          **get_resnet_conv2d_params())
    else:
      residual = conv2d_same(residual, channel_middle, 3, stride=stride,
                             rate=rate, scope='conv2')
    residual = tf.contrib.layers.conv2d(residual, channel_out, [1, 1], stride=1,
                                        rate=rate, scope='conv3',
                                        **get_resnet_conv2d_params(False))

    output = tf.nn.relu(shortcut + residual)

    return output

def resnet_block(inputs, channel_in, channel_out, stride, rate, num_units,
                 scope="", rate_multiple=None):
  '''
  a resnet block
  '''
  assert num_units >= 3, 'num_units should >= 3'
  if (num_units == 3) and (rate_multiple != None) and (len(rate_multiple) == 3):
    # this is block4 used in segmentation net
    with tf.variable_scope(scope):
      with tf.variable_scope('unit_1'):
        net = resnet_block_first_unit(inputs, channel_in, channel_out, rate * rate_multiple[0])
      with tf.variable_scope('unit_2'):
        net = resnet_block_middle_unit(net, channel_out, rate * rate_multiple[1])
      with tf.variable_scope('unit_3'):
        net = resnet_block_last_unit(net, channel_out, stride, rate * rate_multiple[2])
  else:
    with tf.variable_scope(scope):
      with tf.variable_scope('unit_1'):
        net = resnet_block_first_unit(inputs, channel_in, channel_out, rate)
      for i in range(2, num_units):
        with tf.variable_scope('unit_%d' % i):
          net = resnet_block_middle_unit(net, channel_out, rate)
      with tf.variable_scope('unit_%d' % (num_units)):
        net = resnet_block_last_unit(net, channel_out, stride, rate)

  return net

def aspp_layer(inputs, output_stride=16, scope="aspp"):
  '''
  ASPP layer
  '''
  with tf.variable_scope(scope):
    if output_stride == 16:
      rate_ratio = 1
    else:
      rate_ratio = 2

    org_shape = tf.shape(inputs)
    aspp_conv1x1 = tf.contrib.layers.conv2d(inputs, 256, [1, 1],
                                            rate=1, scope='conv1',
                                            **get_resnet_conv2d_params())
    aspp_conv3x3_6 = tf.contrib.layers.conv2d(inputs, 256, [3, 3],
                                              rate=6*rate_ratio, scope='conv3_6',
                                              **get_resnet_conv2d_params())
    aspp_conv3x3_12 = tf.contrib.layers.conv2d(inputs, 256, [3, 3],
                                               rate=12*rate_ratio, scope='conv3_12',
                                               **get_resnet_conv2d_params())
    aspp_conv3x3_18 = tf.contrib.layers.conv2d(inputs, 256, [3, 3],
                                               rate=18*rate_ratio, scope='conv3_18',
                                               **get_resnet_conv2d_params())
    avg_pool = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    avg_pool = tf.contrib.layers.conv2d(avg_pool, 256, [1, 1], scope='conv_avg_pool',
                                        **get_resnet_conv2d_params())
    avg_pool = tf.image.resize_bilinear(avg_pool, [org_shape[1], org_shape[2]])
    concated = tf.concat([aspp_conv1x1, aspp_conv3x3_6, aspp_conv3x3_12, aspp_conv3x3_18, avg_pool],
                         axis=-1)
    output = tf.contrib.layers.conv2d(concated, 256, [1, 1], rate=1, scope='conv_concated',
                                      **get_resnet_conv2d_params())

    return output

def resnet_score_layer(inputs, out_channel, scope='logits'):
  '''
  the last conv layer to output per-pixel score
  '''
  conv = tf.contrib.layers.conv2d(inputs, out_channel, [1, 1], scope=scope,
                                  activation_fn=None,
                                  weights_initializer=tf.truncated_normal_initializer(),
                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                  biases_initializer=tf.zeros_initializer(),
                                  biases_regularizer=None,
                                  normalizer_fn=None)
  return conv
