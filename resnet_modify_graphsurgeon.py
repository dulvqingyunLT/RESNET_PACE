#!/usr/bin/env python3


import onnx_graphsurgeon as gs
import numpy as np
import onnx
import os


def main():

    # graph = gs.import_onnx(onnx.load("/Users/weijunkuang/codes/MA_Examples/onnx_quantize/resnet50_3x224x224_b0.onnx"))
    onnx_model_path = 'resnet50_3x224x224_op12.onnx'
    graph = gs.import_onnx(onnx.load(onnx_model_path))
    tensors = graph.tensors()

    # for index, node in enumerate(graph.nodes):
    #     if node.op == 'Gemm':     
    #         conv_out_0 = gs.Variable(name='conv_out_0', dtype=np.float32, shape=(-1, 1000, 1, 1))
    #         share_fc_0_node = gs.Node(op='Conv', inputs=[flatten_0_out, share_fc_0_weight], outputs=[share_fc_0_out])
    #         graph.nodes.append(share_fc_0_node)

    for index, node in enumerate(graph.nodes):
        if node.op == 'Flatten':
            
            gemm_node = graph.nodes[121]

            np_w = gemm_node.inputs[1].values.reshape(1000, 2048, 1, 1)
            # np_w = gemm_node.inputs[1].values.reshape(10, 2048, 1, 1)
            conv_w = gs.Constant(name='W', values=np_w)

            conv_b = gs.Constant(name='B', values=gemm_node.inputs[2].values)
            conv_out_0 = gs.Variable(name='conv_out_0', dtype=np.float32, shape=(-1, 1000, 1, 1))
            conv_node_0 = gs.Node(op='Conv', name= 'Conv_1000', inputs=[node.inputs[0], conv_w, conv_b], outputs=[conv_out_0])
            graph.nodes.append(conv_node_0)

            out_shape = gs.Constant(name='out_shape', values=np.array([-1, 1000], dtype=np.int64))
            reshape_out = gs.Variable(name='out', dtype=np.float32, shape=(-1, 1000)) 
            reshape_node = gs.Node(op='Reshape', name= 'Reshape_1001', inputs=[conv_out_0,out_shape], outputs=[reshape_out])
            graph.nodes.append(reshape_node)
    
    graph.outputs=[reshape_out]            

    graph.cleanup()
    modified_model = gs.export_onnx(graph)
    onnx.checker.check_model(modified_model)

    modified_model_path = os.path.splitext(onnx_model_path)[0]
    onnx.save(modified_model, modified_model_path+'_modified.onnx')    



if __name__ == '__main__':
    main()




