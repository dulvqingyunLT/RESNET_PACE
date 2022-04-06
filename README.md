# if it is neccessry, change the fix batch input to dynamic batch input, and verify the fp32 and int8 model have the same output.
    onnx_model_path = 'resnet50_3x224x224_op12.onnx'
    qt_onnx_model_path = 'resnet50-v1-12-int8/resnet50-v1-12-int8_dynamic.onnx'

    convert_model_batch_to_dynamic('resnet50-v1-12-int8/resnet50-v1-12-int8.onnx')
    verify_quantized_model(onnx_model_path, qt_onnx_model_path)


# step1. draw every node's output as an output node
    extract_node_info(model_path=qt_onnx_model_path,
            input_shapes = {input:[1,3,224,224]},
            dynamic_input_shape=True
            )


# step2. running the inference in main.py
