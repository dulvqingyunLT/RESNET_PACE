import onnx
from onnx import helper
import onnx_graphsurgeon as gs
from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn.functional as F

import onnxruntime
from inference_helpers import preprocess_one_image,preprocess_func


def clean_graph(model_path):
    model = onnx.load(model_path)
    gs_graph = gs.import_onnx(model)
    gs_graph.outputs=[gs_graph.outputs[0]]
    gs_graph.cleanup()
    modified_model = gs.export_onnx(gs_graph)
    # onnx.checker.check_model(modified_model)
    cleaned_model_path = os.path.splitext(model_path)[0]
    onnx.save(modified_model, cleaned_model_path+'_cleaned.onnx')

def convert_model_batch_to_dynamic(onnx_model_path):
    from onnx.tools import update_model_dims

    onnx_model =  onnx.load(onnx_model_path)
    updated_model = update_model_dims.update_inputs_outputs_dims(onnx_model,
        {'data':[-1,3,224,224]},
        {'resnetv17_dense0_fwd':[-1,1000]})
    
    graph = updated_model.graph
    input_tensor = graph.input[0]
    input_shape = input_tensor.type.tensor_type.shape.dim

    input_tensor_new = onnx.helper.make_tensor_value_info(
        name=input_tensor.name, elem_type=1,
        shape=[-1, input_shape[1].dim_value, input_shape[2].dim_value, input_shape[3].dim_value]
    )
    graph.input.remove(input_tensor)
    graph.input.insert(0,input_tensor_new)

    infered_onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    dynamic_model_name = os.path.splitext(onnx_model_path)[0] + '_dynamic.onnx'
    onnx.save(infered_onnx_model, dynamic_model_name)


def extract_const_shape(nodes):
    """nodes: onnx_graphsurgeon nodes"""
    const_shapes = dict()
    for node in nodes:
        for inp in node.inputs:
            if type(inp).__name__ == 'Constant':
                const_shapes[inp.name] = list(inp.shape)
    return const_shapes


def extract_node_info(model_path, input_shapes=None, input_data=None, dynamic_input_shape=False):
    """
    model: onnx model path
    input_shapes: input shape dict
    input_data: input data dict
    dynamic_input_shape: True if dynamic model
    """
    print(f'Loading ONNX model from {model_path}')
    model = onnx.load(model_path)
    assert(isinstance(model, onnx.ModelProto))
    # onnx.checker.check_model(model_path)

    gs_graph = gs.import_onnx(model)

    node_info = OrderedDict()

    print('Extracting constant shapes')
    const_shapes = extract_const_shape(gs_graph.nodes)

    org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in gs_graph.inputs}
    print('org_inputs',org_inputs)
    org_outputs = [out.name for out in gs_graph.outputs]
    count = 0
    for node in model.graph.node:
        node_info[node.name] = dict()
        attrs = []
        for att in node.attribute:
            attrs.append(str(att).replace("\n"," "))
        node_info[node.name]['op_type'] = node.op_type
        node_info[node.name]['attribute'] = attrs
        node_info[node.name]['inputs'] = dict()
        for inp in node.input:
            if inp in const_shapes:
                node_info[node.name]['inputs'][inp] = const_shapes[inp]
            else:
                node_info[node.name]['inputs'][inp] = [-1]
        node_info[node.name]['outputs'] = dict()
        for output in node.output:
            node_info[node.name]['outputs'][output] = [-1]
            value_info = helper.ValueInfoProto()
            value_info.name = output
            model.graph.output.append(value_info)
            count+=1
    print(f'{count} extra nodes are marked')

    # export intermediate model
    inter_model_path = os.path.splitext(model_path)[0] + '_all_outputs.onnx'
    onnx.save_model(model, inter_model_path)
    print(f'Intermediate model saved at {inter_model_path}')


def get_conv_core_info(onnx_model_path, calibrate_op_types):
    """
    onnx_model_path: onnx model path
    calibrate_op_types: list of op types should be collected.
    """

    model = onnx.load(onnx_model_path)

    graph = gs.import_onnx(model)
    all_nodes = graph.nodes
    # all_tensors = graph.tensors

    all_conv_cores = dict()

    for node in all_nodes:
        if node.op in calibrate_op_types:
            for input_tensor in node.inputs:
                if isinstance(input_tensor, gs.Constant) and len(input_tensor.values.shape) == 4: # 判断是否是卷积核
                    # all_conv_cores[node.name]=input_tensor.values
                    attr = node.attrs
                    attr['values']=input_tensor.values
                    all_conv_cores[node.inputs[0].name] = attr # using input name to identify the conv core params
                    

    # para_path = os.path.splitext(onnx_model_path)[0]
    # np.savez(para_path+'.npz', **all_conv_cores)
    return all_conv_cores


def verify_quantized_model(onnx_model_path, qt_onnx_model_path):
    image_path = '/home/wkuang/data/imagenet-mini/calibration/'

    onnxruntime.set_default_logger_severity(3)
    session = onnxruntime.InferenceSession(onnx_model_path)
    print("session created")
    qt_session = onnxruntime.InferenceSession(qt_onnx_model_path)
    print("quantized_session created")

    gs_graph = gs.import_onnx(onnx.load(onnx_model_path))
    org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in gs_graph.inputs}
    inputs = dict()
    for inp in session.get_inputs():
        inputs[inp.name] = preprocess_func(image_path, height=224, width=224, start_index=0, size_limit=10)
    outputs = [x.name for x in session.get_outputs()]

    print('inputs:',{k:v.shape for k,v in inputs.items()})
    print('outputs names:', outputs)
    ort_outputs = session.run(outputs, inputs)

    ort_result = np.argmax(np.concatenate(ort_outputs,axis=0),axis=-1)


    qt_gs_graph = gs.import_onnx(onnx.load(onnx_model_path))
    qt_org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in qt_gs_graph.inputs}
    qt_inputs = dict()
    for inp in qt_session.get_inputs():
        qt_inputs[inp.name] = preprocess_func(image_path, height=224, width=224, start_index=0, size_limit=10)
    qt_outputs = [x.name for x in qt_session.get_outputs()]

    print('inputs:',{k:v.shape for k,v in qt_inputs.items()})
    print('outputs names:', qt_outputs)


    qt_ort_outputs = qt_session.run(qt_outputs, qt_inputs)
    qt_ort_result = np.argmax(np.concatenate(qt_ort_outputs,axis=0),axis=-1)

    return np.allclose(qt_ort_result,ort_result)


def dot_product_analysis(image_path, weight_dims, img_nums, qt_onnx_model_path, qt_onnx_alloutput_model_path):

    calibrate_op_types = ['QLinearConv']               
    conv_paras = get_conv_core_info(qt_onnx_model_path,calibrate_op_types)

    onnxruntime.set_default_logger_severity(3)
    session = onnxruntime.InferenceSession(qt_onnx_alloutput_model_path)
    print("session created")

    gs_graph = gs.import_onnx(onnx.load(qt_onnx_alloutput_model_path))
    org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in gs_graph.inputs}
    inputs = dict()
    for inp in session.get_inputs():
        # inputs[inp.name] = np.random.random(size=inp.shape).astype(org_inputs[inp.name]['dtype'])
        inputs[inp.name] = preprocess_func(image_path, height=224, width=224, start_index=0, size_limit=img_nums)
    outputs = [x.name for x in session.get_outputs()]

    print('inputs:',{k:v.shape for k,v in inputs.items()})
    print('outputs names:', outputs)
    ort_outputs = session.run(outputs, inputs)
    # conv_paras = np.load('resnet50-v1-12-int8/resnet50-v1-12-int8_dynamic.npz', allow_pickle=True)
    batch_size = ort_outputs[0].shape[0]

    none_zero_input_nums = torch.zeros(batch_size)
    all_elem_input_nums = 0

    for input_name in conv_paras.keys() :
        weight_info = conv_paras[input_name]
        inp = ort_outputs[outputs.index(input_name)]
        # batch_size = inp.shape[0]
        kernel = weight_info['kernel_shape']
        padding = weight_info['pads']
        stride = weight_info['strides']
        weight__ = weight_info['values']
        output_channel, input_channel, _, _ = weight__.shape
        weight__ = torch.from_numpy(weight__)
        inp_unf_ = torch.nn.functional.unfold(torch.from_numpy(inp.astype(np.float16)), kernel_size=kernel, padding=padding[0], stride=stride[0])  #[b, c*k*k, L]#c*k*k表示有多少个局部块，L表示局部块的大小      
        inp_unf_=inp_unf_.transpose(1, 2).type(torch.uint8)  #[b, L, c*k*k]  

        weight_ = weight__.view(output_channel, -1).t() #[out_c,c*k*k]-->[c*k*k, out_c]

        repeats = inp_unf_.size(-1) // weight_dims
        remainder = inp_unf_.size(-1) % weight_dims
        repeats = repeats+1 if remainder!=0 else repeats

        dim=(0, weight_dims*repeats - weight_.size(0), 0, 0) #左右上下， 填右边
        pad_tensor_inp=F.pad(inp_unf_,dim,"constant",value=0)
 
        inp_unf = pad_tensor_inp.contiguous().view([batch_size, inp_unf_.size(1), repeats, -1])

        dim=(0, 0, 0, weight_dims*repeats - weight_.size(0)) #左右上下， 填下边
        pad_tensor_weight=F.pad(weight_,dim,"constant",value=0)

        weight = pad_tensor_weight.permute(1,0).reshape(weight_.size(1), repeats, -1).permute(2,1,0)
        temp_out = torch.zeros([batch_size, inp_unf_.size(1), weight_.size(1)])
        inp_unf_sum = inp_unf.sum(dim=-1, keepdim=False)#通过求和判断是否全零

        none_zero_input_nums += inp_unf_sum.count_nonzero(dim=(1,2))//64 # 只代表输入中非全零输入vector的数量，不代表矩阵乘的次数
        all_elem_input_nums += inp_unf_sum.shape[1]*inp_unf_sum.shape[2]//64

    return  none_zero_input_nums/all_elem_input_nums


if __name__=='__main__':


# if it is neccessry, change the fix batch input to dynamic batch input, and verify the fp32 and int8 model have the same output.
    onnx_model_path = 'resnet50_3x224x224_op12.onnx'
    qt_onnx_model_path = 'resnet50-v1-12-int8/resnet50-v1-12-int8_dynamic.onnx'

    # convert_model_batch_to_dynamic('resnet50-v1-12-int8/resnet50-v1-12-int8.onnx')
    # verify_quantized_model(onnx_model_path, qt_onnx_model_path)


#step1. draw every node's output as an output node
    # extract_node_info(model_path=qt_onnx_model_path,
    #         input_shapes = {input:[1,3,224,224]},
    #         dynamic_input_shape=True
    #         )


# step2. running the inference

    image_path = '/home/wkuang/data/imagenet-mini/calibration/'
    # image_path = '/home/wkuang/data/imagenet-mini/calibration/ILSVRC2012_val_00000584.JPEG'
    qt_onnx_alloutput_model_path = 'resnet50-v1-12-int8/resnet50-v1-12-int8_dynamic_all_outputs.onnx'


    ratios_dim64 = dot_product_analysis(image_path, 64, 50, qt_onnx_model_path, qt_onnx_alloutput_model_path)

    ratios_dim16 = dot_product_analysis(image_path, 16, 50, qt_onnx_model_path, qt_onnx_alloutput_model_path)

    print('end')


    
