'''
draw masks on the image for voc labels
'''
import platform
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def get_palette(palette_image, class_name_list='cityscapes_9cls', include_color0=False):
  '''
  return the class name and the palette color
  '''
  #load palette
  palette = palette_image.getpalette()
  if class_name_list.lower() == 'cityscapes_9cls':
    labels = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
  else:
    assert False, 'unknown class name list'

  rtn = [{}]
  if include_color0:
    palette_idx = -3
  else:
    palette_idx = 0
  for label in labels:
    palette_idx += 3
    color_r = palette[palette_idx]
    color_g = palette[palette_idx+1]
    color_b = palette[palette_idx+2]
    rtn.append({
        'rgb': (color_r, color_g, color_b),
        'label': label
    })

  return rtn

def get_mask_colors(mask_img):
  '''
  get mask colors
  '''
  mask_arr = np.array(mask_img)
  mask_arr = np.array(mask_arr, dtype=int)
  mask_arr = np.reshape(mask_arr, [len(mask_arr) * len(mask_arr[0])])
  mask_arr = np.unique(mask_arr)
  return list(mask_arr)

def get_legends(img, colors, palette):
  '''
  draw legends
  '''
  #rtn
  rtn = []
  rtn_lines = 1

  #init draw
  draw = ImageDraw.Draw(img)
  if platform.system() == 'Windows':
    font = ImageFont.truetype('arial.ttf', 15)
  elif platform.system() == 'Linux':
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
  else:
    assert False, 'not supported platform'

  #position
  x_len = 15
  y_len = 15
  x_pos = 7
  y_pos = 7

  #offset
  x_off = 0
  for color in colors:
    if color == 0:
      continue
    text_size = draw.textsize(palette[color]['label'], font)[0]
    if text_size > x_off:
      x_off = text_size
  x_off += x_len + 20
  y_off = y_len + 7

  #generate legends
  for color in colors:
    if color == 0:
      continue
    if x_pos + x_off >= img.width:
      x_pos = 7
      y_pos += y_off
      rtn_lines += 1
    rtn.append({
        'rect_pos': [x_pos, y_pos, x_pos + x_len, y_pos + y_len],
        'label_pos': (x_pos + x_len + 5, y_pos),
        'fill': palette[color]['rgb'],
        'label': palette[color]['label']
    })
    x_pos += x_off

  return rtn, (7 + y_len) * rtn_lines + 7

def draw_legend(img, legends):
  '''
  draw legend
  '''
  #init draw
  draw = ImageDraw.Draw(img)
  if platform.system() == 'Windows':
    font = ImageFont.truetype('arial.ttf', 15)
  elif platform.system() == 'Linux':
    font = ImageFont.truetype('DejaVuSans.ttf', 14)
  else:
    assert False, 'not supported platform'

  for legend in legends:
    draw.rectangle(legend['rect_pos'], fill=legend['fill'])
    draw.text(legend['label_pos'], legend['label'],
              fill=(255, 255, 255), font=font)

def combine(img_file, mask_file, class_name_list='VOC', include_color0=False, has_legend=True):
  '''
  combine the jpg image and the mask into a single jpg image
  '''
  #load image
  img = Image.open(img_file)
  mask_p = Image.open(mask_file)
  mask_index = np.array(mask_p)
  mask_alpha = np.where(np.equal(mask_index, 0), 0, 180)
  mask_alpha = Image.fromarray(mask_alpha.astype(np.uint8), mode='L')

  #init palette
  palette = get_palette(mask_p, class_name_list, include_color0)

  #add_legend
  if has_legend:
    mask_colors = get_mask_colors(mask_p)
    (legends, off_height) = get_legends(img, mask_colors, palette)
    rtn_img = Image.new('RGBA', (img.width, img.height + off_height))
    draw_legend(rtn_img, legends)
  else:
    off_height = 0
    rtn_img = Image.new('RGBA', (img.width, img.height + off_height))

  #combime
  img = img.convert('RGBA')
  img.putalpha(150)
  rtn_img.alpha_composite(img, (0, off_height))
  mask = mask_p.convert('RGBA')
  mask.putalpha(mask_alpha)
  rtn_img.alpha_composite(mask, (0, off_height))

  return rtn_img.convert('RGB')
