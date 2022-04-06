
import onnx
from onnx import helper, TensorProto, ModelProto
from onnx import onnx_pb as onnx_proto
import os
import itertools
import torch
from torchvision.models.resnet import resnet50


def torch_to_onnx():

    model = resnet50(pretrained=True, progress=True)
    model.eval()

    x = torch.randn(1, 3, 224, 224)   # 生成张量

    export_onnx_file = "resnet50_3x224x224_op12.onnx"# 目的ONNX文件名
    torch.onnx.export(model,
                        x,
                        export_onnx_file,
                        opset_version=12,
                        do_constant_folding=True,	# 是否执行常量折叠优化
                        input_names=["input"],	# 输入名
                        output_names=["output"],	# 输出名
                        dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                        "output":{0:"batch_size"}}
    )                                    

def select_tensors_to_calibrate(model, op_types_to_calibrate):
    '''
    select all quantization_candidates op type nodes' input/output tensors. 
    returns:
        tensors (set): set of tensor name.
        value_infos (dict): tensor name to value info.
    '''
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    value_infos.update({ot.name: ot for ot in model.graph.output})
    value_infos.update({it.name: it for it in model.graph.input})
    value_infos.update({ini.name: ini for ini in model.graph.initializer})
    # initializer = set(init.name for init in model.graph.initializer)

    tensors_to_calibrate = set()
    # tensor_type_to_calibrate = set([TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.INT8])

    for node in model.graph.node:
        if node.op_type in op_types_to_calibrate:
            # for tensor_name in itertools.chain(node.input, node.output):
            for tensor_name in node.output:
                if tensor_name in value_infos.keys():
                    # vi = value_infos[tensor_name]
                    # if vi.type.HasField('tensor_type') and (tensor_name not in initializer):
                    tensors_to_calibrate.add(tensor_name)

    return tensors_to_calibrate, value_infos


def augment_graph(onnx_model_path, op_types_to_calibrate):
    '''
    make all quantization_candidates op type nodes as part of the graph output.
    :return: augmented ONNX model
    '''
    onnx_model = onnx.load(onnx_model_path)
    model = onnx_proto.ModelProto()
    model.CopyFrom(onnx_model)
    model = onnx.shape_inference.infer_shapes(model)

    added_nodes = []
    added_outputs = []
    tensors, value_infos = select_tensors_to_calibrate(model, op_types_to_calibrate) 

    for tensor in tensors:
        added_outputs.append(value_infos[tensor])

    model.graph.node.extend(added_nodes)
    model.graph.output.extend(added_outputs)
    augmented_model_path = os.path.splitext(onnx_model_path)[0]
    onnx.save(model, augmented_model_path+'.onnx')
    

