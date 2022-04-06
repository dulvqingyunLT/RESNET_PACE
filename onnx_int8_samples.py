
import sys
import os
from onnxruntime.quantization.calibrate import CalibrationMethod

import onnxsim
from onnxruntime.quantization import quantize_qat, quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization.quant_utils import QuantFormat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.onnx_quantization_utils.data_read import CustomizeCalibrationDataReader
from utils.onnx_quantization_utils.datasets import object_detection_preprocess_func


def test_build_onnx_int8_resnet50(onnx_model_path):  

    model_fp32_path = onnx_model_path
    quant_out_model_path = os.path.splitext(onnx_model_path)[0] + '_int8.onnx'
    
    calibration_dataset = '/home/wkuang/data/imagenet-mini/calibration'

    data_reader = CustomizeCalibrationDataReader(calibration_dataset,
                                    width=224,
                                    height=224,
                                    start_index=0,
                                    end_index=200,
                                    stride=10, # >=batch_size
                                    batch_size=10,
                                    input_name='input', 
                                    # model_path=simplifed_model_fp32_path,
                                    preprocess_func=object_detection_preprocess_func
                                    )

    op_types_to_quantize = [ 'Conv', 'MatMul', 'Add', 'Mul', 'Relu', 'Clip', 'LeakyRelu', 'Sigmoid', 'MaxPool',
                            'GlobalAveragePool', 'Split', 'Pad', 
                            'Reshape', 
                            'Transpose',
                            'Squeeze',
                            'Unsqueeze',
                            'Resize',
                            'AveragePool',
                            'Concat',
                            'Gather',
                            'EmbedLayerNormalization',
                            'Flatten',
                            'QuantizeLinear',
                            'QLinearConv',
    ]
    quantize_static(model_input=model_fp32_path, 
                                    model_output=quant_out_model_path,
                                    calibration_data_reader=data_reader,
                                    quant_format=QuantFormat.QOperator,
                                    op_types_to_quantize=op_types_to_quantize,
                                    per_channel=False,
                                    reduce_range=False,
                                    activation_type=QuantType.QUInt8,
                                    weight_type=QuantType.QUInt8,
                                    nodes_to_quantize=[],
                                    nodes_to_exclude=[],
                                    optimize_model=False,
                                    use_external_data_format=False,
                                    calibrate_method=CalibrationMethod.Entropy,
                                    extra_options={}
                                    )
    print("quantization end!")


if __name__ == '__main__':

    test_build_onnx_int8_resnet50('resnet50_3x224x224_op12.onnx')



