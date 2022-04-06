
from onnxruntime.quantization import create_calibrator, write_calibration_table, CalibrationMethod
from onnx_quantization_utils.data_read import CustomizeCalibrationDataReader
from onnx_quantization_utils.datasets import ObjectDetectionEvaluator




def get_calibration_table(model_path, augmented_model_path, calibration_dataset):

    calibrator = create_calibrator(model_path, [], augmented_model_path=augmented_model_path)
    calibrator.set_execution_providers(['TensorrtExecutionProvider'])

    # DataReader can handle dataset with batch or serial processing depends on its implementation
    # Following examples show two different ways to generate calibration table
    '''
    1. Use serial processing
    
    We can use only one DataReader to do serial processing, however,
    some machines don't have sufficient memory to hold all dataset images and all intermediate output.
    So let multiple DataReader do handle different stride of dataset one by one.
    DataReader will use serial processing when batch_size is 1.
    '''

    # total_data_size = len(os.listdir(calibration_dataset))
    # start_index = 0
    # stride = 2000
    # for i in range(0, total_data_size, stride):
    #     data_reader = YoloV3DataReader(calibration_dataset,
    #                                    start_index=start_index,
    #                                    end_index=start_index + stride,
    #                                    stride=stride,
    #                                    batch_size=1,
    #                                    model_path=augmented_model_path)
    #     calibrator.collect_data(data_reader)
    #     start_index += stride
    '''
    2. Use batch processing (much faster)
    
    Batch processing requires less memory for intermediate output, therefore let only one DataReader to handle dataset in batch. 
    However, if encountering OOM, we can make multiple DataReader to do the job just like serial processing does. 
    DataReader will use batch processing when batch_size > 1.
    '''

    data_reader = CustomizeCalibrationDataReader(calibration_dataset, stride=500, batch_size=10, model_path=augmented_model_path)
    calibrator.collect_data(data_reader)

    write_calibration_table(calibrator.compute_range())
    print('calibration table generated and saved.')


def get_prediction_evaluation(model_path, validation_dataset, providers):
    data_reader = CustomizeCalibrationDataReader(validation_dataset,
                                   stride=1000,
                                   batch_size=20,
                                   model_path=model_path,
                                   is_evaluation=True)
    evaluator = ObjectDetectionEvaluator(model_path, data_reader, providers=providers)

    evaluator.predict()
    result = evaluator.get_result()

    annotations = './annotations/instances_val2017.json'
    evaluator.evaluate(result, annotations)