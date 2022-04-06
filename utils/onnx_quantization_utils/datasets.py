# from onnxruntime.quantization import create_calibrator, write_calibration_table, CalibrationMethod
# from preprocessing import yolov3_preprocess_func, yolov3_preprocess_func_2, yolov3_variant_preprocess_func, yolov3_variant_preprocess_func_2
import random
import onnxruntime
from argparse import Namespace
import os
import numpy as np
import cv2
import copy
from onnxruntime.quantization import CalibrationDataReader

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



def object_detection_preprocess_func(images_folder, height, width, start_index=0, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''

    image_names = os.listdir(images_folder)
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

    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list


def segmentation_preprocess_func(images_folder, height, width, start_index=0, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''

    image_names = []

    npy_cal_map = os.path.join(images_folder, 'npy_cal_map.txt')
    with open(npy_cal_map, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            image_names.append(line)
    
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
        image_filepath = images_folder + '/fp32/' + image_name + '.npy'
        # model_image_size = [4, height, width]

        image_data = np.load(image_filepath)
        # img = cv2.imread(image_filepath)
        # image_data = preprocess_one_image(img, model_image_size) 

        # add alpha channel to fit the unet
        

        image_data = np.ascontiguousarray(image_data)
        image_data = np.expand_dims(image_data, 0)
        unconcatenated_batch_data.append(image_data)
        # _height, _width, _ = img.shape
        # image_size_list.append(img.shape[0:2])  # img.shape is h, w, c
        image_size_list.append(np.array([image_data.shape[2], image_data.shape[3]], dtype=np.float32).reshape(1, 2))

    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data, batch_filenames, image_size_list


class ObjectDetectionEvaluator:
    def __init__(self,
                 model_path,
                 data_reader: CalibrationDataReader,
                 width=384,
                 height=384,
                 providers=["CUDAExecutionProvider"],
                 ground_truth_object_class_file="./coco-object-categories-2017.json",
                 onnx_object_class_file="./onnx_coco_classes.txt"):
        '''
        :param model_path: ONNX model to validate 
        :param data_reader: user implemented object to read in and preprocess calibration dataset
                            based on CalibrationDataReader Interface

        '''
        self.model_path = model_path
        self.data_reader = data_reader
        self.width = width
        self.height = height
        self.providers = providers
        self.class_to_id = {}  # object class -> id
        self.onnx_class_list = []
        self.prediction_result_list = []
        self.identical_class_map = {
            "motorbike": "motorcycle",
            "aeroplane": "airplane",
            "sofa": "couch",
            "pottedplant": "potted plant",
            "diningtable": "dining table",
            "tvmonitor": "tv"
        }

        f = open(onnx_object_class_file, 'r')
        lines = f.readlines()
        for c in lines:
            self.onnx_class_list.append(c.strip('\n'))

        self.generate_class_to_id(ground_truth_object_class_file)
        print(self.class_to_id)

        self.session = onnxruntime.InferenceSession(model_path, providers=providers)

    def generate_class_to_id(self, ground_truth_object_class_file):
        with open(ground_truth_object_class_file) as f:
            import json
            classes = json.load(f)

        for c in classes:
            self.class_to_id[c["name"]] = c["id"]

    def set_data_reader(self, data_reader):
        self.data_reader = data_reader

    def get_result(self):
        return self.prediction_result_list

    def set_bbox_prediction(self, boxes, scores, indices, is_batch, image_id, image_id_batch):
        out_boxes, out_scores, out_classes, out_batch_index = [], [], [], []

        for idx_ in indices:
            out_classes.append(idx_[1])
            out_batch_index.append(idx_[0])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])

        for i in range(len(out_classes)):
            out_class = out_classes[i]
            class_name = self.onnx_class_list[int(out_class)]
            if class_name in self.identical_class_map:
                class_name = self.identical_class_map[class_name]
            id = self.class_to_id[class_name]

            bbox = [out_boxes[i][1], out_boxes[i][0], out_boxes[i][3], out_boxes[i][2]]
            bbox_yxhw = [
                out_boxes[i][1], out_boxes[i][0], out_boxes[i][3] - out_boxes[i][1], out_boxes[i][2] - out_boxes[i][0]
            ]
            bbox_yxhw_str = [
                str(out_boxes[i][1]),
                str(out_boxes[i][0]),
                str(out_boxes[i][3] - out_boxes[i][1]),
                str(out_boxes[i][2] - out_boxes[i][0])
            ]
            score = str(out_scores[i])
            coor = np.array(bbox[:4], dtype=np.int32)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

            if is_batch:
                image_id = image_id_batch[out_batch_index[i]]
            self.prediction_result_list.append({
                "image_id": int(image_id),
                "category_id": int(id),
                "bbox": bbox_yxhw,
                "score": out_scores[i]
            })

    def predict(self):
        session = self.session

        outputs = []

        # If you decide to run batch inference, please make sure all input images must be re-sized to the same shape.
        # Which means the bounding boxes from groun truth annotation must to be adjusted accordingly, otherwise you will get very low mAP results.
        # Here we simply choose to run serial inference.
        if self.data_reader.get_batch_size() > 1:
            # batch inference
            print("Doing batch inference...")

            image_id_list = []
            image_id_batch = []
            while True:
                inputs = self.data_reader.get_next()
                if not inputs:
                    break
                image_id_list = inputs["image_id"]
                del inputs["image_id"]
                image_id_batch.append(image_id_list)
                outputs.append(session.run(None, inputs))

                for index in range(len(outputs)):
                    output = outputs[index]
                    boxes = output[0]
                    indices = output[1]
                    # scores = output[1]
                    # indices = output[2]

                    # self.set_bbox_prediction(boxes, scores, indices, True, None, image_id_batch[index])
        else:
            # serial inference
            while True:
                inputs = self.data_reader.get_next()
                if not inputs:
                    break

                image_id = inputs["image_id"]
                del inputs["image_id"]

                output = session.run(None, inputs)

                boxes = output[0]
                scores = output[1]
                indices = output[2]

                self.set_bbox_prediction(boxes, scores, indices, False, image_id, None)

    def evaluate(self, prediction_result, annotations):
        # calling coco api
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        annFile = annotations
        cocoGt = COCO(annFile)

        resFile = prediction_result
        cocoDt = cocoGt.loadRes(resFile)

        imgIds = sorted(cocoGt.getImgIds())

        # running evaluation
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()