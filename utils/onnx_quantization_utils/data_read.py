from onnxruntime.quantization import CalibrationDataReader
import onnxruntime
#from onnx_quantization_utils.datasets import object_detection_preprocess_func, parse_annotations
import os
import numpy as np



class CustomizeCalibrationDataReader(CalibrationDataReader):
    '''
    this class is used to read image data for calibration
    '''
        
    def __init__(self,
                 calibration_image_folder,
                 width=384,
                 height=384,
                 start_index=0,
                 end_index=0,
                 stride=1,
                 batch_size=1,
                #  model_path='augmented_model.onnx',
                 input_name='input',
                #  is_evaluation=False,
                #  annotations='/home/ubuntu/data/coco//annotations/instances_val2017.json',
                 preprocess_func=None):
        '''
        :param calibration_image_folder: image path for calibration
        :param widht: input image width for onnx model
        :param height: input image height for onnx model
        :param start_index: start index of the image list from calibration_image_folder
        :param end_index: end index of the image list from calibration_image_folder
        :param stride: suggest set to batch_size,could be larger than batch_size
        :param batch_size: batch size of input to model
        :param input_name: input of onnx model, list of strings if have multi input
        :param preprocess_func: image preprocess function before feed to model
        '''

        CalibrationDataReader.__init__(self)
        self.image_folder = calibration_image_folder 
        # self.model_path = model_path
        self.preprocess_flag = True
        self.enum_data_dicts = iter([])
        self.width = width
        self.height = height
        self.start_index = start_index
        self.end_index = len(os.listdir(calibration_image_folder)) if end_index == 0 else end_index
        self.stride = stride if stride >= 1 else 1  # stride must > 0
        self.batch_size = batch_size
        # self.is_evaluation = is_evaluation

        self.input_name = input_name
        # self.img_name_to_img_id = parse_annotations(annotations)
        self.preprocess_func = preprocess_func

    def get_batch_size(self):
        return self.batch_size

    def get_dataset_size(self):
        return len(os.listdir(self.image_folder))
        # return 100

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            return iter_data

        self.enum_data_dicts = None
        if self.start_index < self.end_index:
            if self.batch_size == 1:
                data = self.load_serial()
            else:
                data = self.load_batches()

            self.start_index += self.stride
            self.enum_data_dicts = iter(data)

            return next(self.enum_data_dicts, None)
        else:
            return None

    def load_serial(self):
        width = self.width
        height = self.height
        nchw_data_list, filename_list, image_size_list = self.preprocess_func(self.image_folder, height, width,
                                                                                self.start_index, self.stride)
        input_name = self.input_name

        print("Start from index %s ..." % (str(self.start_index)))
        data = []
        for i in range(len(nchw_data_list)):
            nhwc_data = nchw_data_list[i]
            file_name = filename_list[i]
            # data.append({input_name: nhwc_data, "image_shape": image_size_list[i]})
            data.append({input_name: nhwc_data}) 
        return data

    def load_batches(self):
        width = self.width
        height = self.height
        batch_size = self.batch_size
        stride = self.stride
        input_name = self.input_name

        for index in range(0, stride, batch_size):
            start_index = self.start_index + index
            print("Load batch from index %s ..." % (str(start_index)))
            nchw_data_list, filename_list, image_size_list = self.preprocess_func(self.image_folder, height, width,
                                                                                    start_index, batch_size)

            if nchw_data_list.size == 0:
                break

            nchw_data_batch = []
            # image_id_batch = []
            batches = []

            for i in range(len(nchw_data_list)):
                nhwc_data = np.squeeze(nchw_data_list[i], 0)
                nchw_data_batch.append(nhwc_data)
            batch_data = np.concatenate(np.expand_dims(nchw_data_batch, axis=0), axis=0)
            print(batch_data.shape)
            # data = {input_name: batch_data, "image_shape": np.asarray([[416, 416]], dtype=np.float32)}
            data = {input_name: batch_data}

            batches.append(data)

        return batches


