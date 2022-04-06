import onnxruntime
from argparse import Namespace
import os
import numpy as np
import cv2
import copy
import random

def parse_annotations(filename):
    import json
    annotations = {}
    with open(filename, 'r') as f:
        annotations = json.load(f)

    img_name_to_img_id = {}
    for image in annotations["images"]:
        file_name = image["file_name"]
        img_name_to_img_id[file_name] = image["id"]

    return img_name_to_img_id

def preprocess_one_image(img, dst_shape=[3, 512, 512], mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], backend='cv2'):
  '''Pre-processing a image with resize and normalization, non inplace modification
  img: imagepath or PIL.Image or 2D/3D np.array. If backend='PIL', input order is HWC-RGB; if backend='cv2', input order is HWC-BGR
  backend: 'PIL' or 'cv2'
  channels: int, desired number of channels, usually 1 or 3
  dst_shape: shape of object image, (channel, height, width)
  Returns:
      img_data: 3D np.array of fp32, the order is CHW-BGR
  '''
  dst_channels = dst_shape[0]
  height, width = dst_shape[1], dst_shape[2]
  data = copy.deepcopy(img)
  if backend == 'PIL':
    from PIL import Image
    if isinstance(data, str):
      data = Image.open(data)
    elif isinstance(data, np.ndarray):
      data = Image.fromarray(data)
    data = data.resize((width, height), Image.ANTIALIAS)
    data = np.asarray(data).astype(np.float32)
    if len(data.shape) == 2:
      data = np.stack([data] * dst_channels) # add dimension of channel
      print(f'accept grayscale image, preprocess to {dst_channels} channels')
    else:
      data = data.transpose([2, 0, 1]) # to CHW
    mean_vec = np.array(mean)
    stddev_vec = np.array(std)
    assert data.shape[0] == dst_channels
    for i in range(data.shape[0]):
      data[i, :, :] = (data[i, :, :] - mean_vec[i]) / stddev_vec[i]
  elif backend == 'cv2':
    import cv2
    if isinstance(data, str):
      data = cv2.imread(data)
    elif not isinstance(data, np.ndarray):
      raise TypeError('input must be image path or np.array for backend cv2')
    if len(data.shape) == 2:
      data = np.stack([data] * dst_channels, axis=-1) # add dimension of channel
      print(f'accept grayscale image, preprocess to {dst_channels} channels')
    data = cv2.resize(data, (width, height))
    data = data.astype(np.float32)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    mean = np.array([mean], dtype=np.float64)
    inv_std = 1 / np.array([std], dtype=np.float64)
    cv2.subtract(data, mean, data)
    cv2.multiply(data, inv_std, data)
    data = data.transpose(2, 0, 1) # to CHW
  return data



def preprocess_func(images_folder, height, width, start_index=0, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    # random.shuffle(image_names)
    if start_index >= len(image_names):
        return np.asanyarray([]), np.asanyarray([]), np.asanyarray([])
    elif size_limit > 0 and len(image_names) >= size_limit:
        end_index = start_index + size_limit
        if end_index > len(image_names):
            end_index = len(image_names)

        batch_filenames = [image_names[i] for i in range(start_index, end_index)]
    else:
        batch_filenames = image_names

    unconcatenated_batch_data = []
    image_size_list = []

    print(batch_filenames)
    print("size: %s" % str(len(batch_filenames)))

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        model_image_size = [3, height, width]

        img = cv2.imread(image_filepath)
        image_data = preprocess_one_image(img, model_image_size) 

        # add alpha channel to fit the unet

        image_data = np.ascontiguousarray(image_data)
        image_data = np.expand_dims(image_data, 0)
        unconcatenated_batch_data.append(image_data)
        _height, _width, _ = img.shape
        # image_size_list.append(img.shape[0:2])  # img.shape is h, w, c
        image_size_list.append(np.array([img.shape[0], img.shape[1]], dtype=np.float32).reshape(1, 2))

    batch_data = np.concatenate(unconcatenated_batch_data, axis=0)
    # return batch_data, batch_filenames, image_size_list
    return batch_data

