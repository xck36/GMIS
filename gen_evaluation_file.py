'''
generate evaluation file for Cityscapes.
'''
import numpy as np
from PIL import Image
import utilities.get_default_palette as get_default_palette

BINS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

def get_mode(arr, num=1):
  '''
  get mode
  '''
  assert num <= 8
  hist = np.histogram(arr, bins=BINS)
  hist = hist[0]
  cur_max = np.max(hist)
  label_list = []
  ratio = []
  for _ in range(num):
    max_value = np.max(hist)
    max_index = np.argmax(hist)
    if max_value > cur_max*0.2:
      label_list.append(max_index)
      ratio.append(max_value/cur_max)
      hist[max_index] = 0
    else:
      break
  return label_list, ratio

def gen_evaluation_file(semantic_mask, instance_mask, confidence_array,
                        instance_mask_path, txt_path, file_name, additive=False,
                        slide_semantic=False, mask_label=None, mask_idx=0, result_path=None):
  '''
  generate evaluation file for cityscapes
  '''
  assert result_path, 'result_path must be specified'
  result_path = result_path + '/'
  _ids = np.array(['0', '24', '25', '26', '27', '28', '31', '32', '33'])
  instance_list = np.unique(instance_mask)

  refined_instance_mask = np.copy(instance_mask)
  if slide_semantic:
    semantic_part_idx = mask_label == (mask_idx + 1)
  file_op = 'w'
  ins_prefix = str(mask_idx)
  if additive:
    file_op = 'a'

  with open(result_path + txt_path + file_name + '.txt', file_op) as file:
    for i in range(instance_list.shape[0]):
      if instance_list[i] == 0:
        continue

      ins_mask = instance_mask == instance_list[i]
      instance_semantic_label = semantic_mask[ins_mask]
      instance_semantic_label = instance_semantic_label[instance_semantic_label != 0]

      if instance_semantic_label.size == 0:
        continue

      labels, ratios = get_mode(instance_semantic_label, 4)

      for label_i, label in enumerate(labels):
        if label == 0:
          continue

        _confidence = confidence_array[i] * ratios[label_i]

        _id = _ids[label]
        if slide_semantic:
          if not np.isin(label, np.arange(9)[semantic_part_idx]):
            refined_instance_mask[ins_mask] = 0
            continue

        curr_ins = ins_prefix + '_%d_%d.png' % (i, label_i)
        _result_mask_path = instance_mask_path + file_name + curr_ins
        _result_mask = ins_mask.astype('uint8')
        _result_mask_image = Image.fromarray(_result_mask, mode='P')
        mask_palette = get_default_palette.get_default_palette()
        _result_mask_image.putpalette(mask_palette)
        _result_mask_image.save(result_path + _result_mask_path)

        file.write('.' + _result_mask_path)
        file.write(' ')
        file.write(_id)
        file.write(' ')
        file.write(str(_confidence))
        file.write('\n')

  return refined_instance_mask
