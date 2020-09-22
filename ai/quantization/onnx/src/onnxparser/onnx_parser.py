
from collections import OrderedDict
import logging
import logging.config

# onnx lib
import onnx
import onnxruntime
from onnx import helper, shape_inference
from onnxparser._transformers import ConvAddFuser, ConstantsToInitializers

import config as cfg
from onnxparser._graph import Graph

logging.config.fileConfig(cfg.log_cfg_path)

transformers = [
    ConstantsToInitializers(),
    ConvAddFuser(),
]

class OnnxPareser(object):
    conv_type_ = ['Conv', 'ConvTranspose']

    def __init__(self,
                 model_path,
                 new_model_path,
                 quanti_layer_type,
                 middle_layer_output_shape,
                 merge_layer_type,
                 merge_layer_indexs,
                 merge_layer_shapes):
        super().__init__()
        self.name = 'OnnxParser'
        self.model_path_ = model_path
        self.graph = self.getGraph(model_path)
        self.__new_model_path_ = new_model_path
        self.__quanti_layer_type_ = quanti_layer_type
        self.__middle_layer_output_shape_ = middle_layer_output_shape
        self.__merge_layer_type_ = merge_layer_type
        self.__merge_layer_indexs_ = merge_layer_indexs
        self.__merge_layer_shapes_ = merge_layer_shapes

        # 获取中间需要量化的层的层名称
        self.middle_layers_name = OrderedDict()
        # 获取需要合并量化参数的网络层的层名称
        self.merge_layers_ = OrderedDict()
        nodes = self.graph.nodes
        for i in range(len(nodes)):
            # 判断是否为卷积需要做量化
            if nodes[i].op_type in self.conv_type_:
                Success, BN_layers_index = \
                    self.find_layer_index_by_name(nodes[i].outputs[0], nodes)
                if Success and nodes[BN_layers_index].op_type == 'BatchNormalization':
                    Success_, Relu_layers_index = \
                        self.find_layer_index_by_name(nodes[BN_layers_index].outputs[0], nodes)
                    if Success_ and nodes[Relu_layers_index].op_type in self.__quanti_layer_type_:
                        self.middle_layers_name[nodes[i].name] = nodes[Relu_layers_index].name
                    else:
                        raise ValueError("Err: don't support this layer type {}"
                                         .format(nodes[i].op_type))
                    del Success_, Relu_layers_index
                else:
                    self.middle_layers_name[nodes[i].name] = nodes[i].name
                    logging.warning("Warning: only quantization conv {}".format(nodes[i].name))
                del Success, BN_layers_index
            elif nodes[i].op_type in self.__merge_layer_type_:
                # 查找需要合并量化参数的层，即该层的输入和输出的量化尺度统一
                input_nodes_name = nodes[i].inputs
                merge_layers = []
                for node_name in input_nodes_name:
                    print(node_name)
                    tmp_name = node_name

                    while True:

                        Success, layer_index = self.find_layer_index_by_current_name(tmp_name, nodes)

                        if Success and nodes[layer_index].op_type in self.conv_type_:
                            merge_layers.append(nodes[layer_index].name)
                            break
                        elif Success and nodes[layer_index].op_type in self.__merge_layer_type_:
                            merge_layers.append(nodes[layer_index].name)
                            break
                        elif not Success:
                            break
                        else:
                            tmp_name = nodes[layer_index].inputs[0]
                        del Success, layer_index

                self.merge_layers_[nodes[i].name] = merge_layers
                del merge_layers, input_nodes_name

                print(nodes[i].op_type)
                pass
            else:
                logging.warning("Warning: don't support this layer type: {} quantization"
                                .format(nodes[i].op_type))
        del nodes
        print(self.middle_layers_name)

        self.model = onnx.load(self.model_path_)
        self.__insert_output_node_to_model(self.model,
                                           self.__new_model_path_,
                                           self.middle_layers_name,
                                           self.__middle_layer_output_shape_,
                                           self.__merge_layer_indexs_,
                                           self.__merge_layer_shapes_)

        # del self.model, self.graph
        self.model = onnx.load(self.__new_model_path_)
        self.graph = self.getGraph(self.__new_model_path_)
        self.__sess_ = onnxruntime.InferenceSession(self.__new_model_path_)

        self.output_layers_name = []
        self.input_layers_name = []

        # 获取模型输出层
        for i in range(len(self.graph.outputs)):
            logging.debug("layer name: {}, shape: {}".format(self.graph.outputs[i][0], str(self.graph.outputs[i][2])))
            self.output_layers_name.append(self.graph.outputs[i][0])

        # 获取输入层
        for i in range(len(self.graph.inputs)):
            logging.debug("layer name: {}, shape: {}".format(self.graph.inputs[i][0], self.graph.inputs[i][2]))
            self.input_layers_name.append(self.graph.inputs[i][0])

        # 获取卷积的权重数据
        self.weights = OrderedDict()
        for i in range(len(self.graph.nodes)):
            node = self.graph.nodes[i]
            if node.op_type == self.conv_type_[0]:
                # logging.debug(i)
                self.weights[node.name] = list(node.input_tensors.values())[0]
            elif node.op_type == self.conv_type_[1]:
                self.weights[node.name] = list(node.input_tensors.values())[0].transpose(1, 0, 2, 3)
                # self.weights[node.name] = list(node.input_tensors.values())[0]
            del node

        logging.debug("parser onnx model success")

    def __del__(self):
        del self.name
        del self.graph
        del self.input_layers_name
        del self.model
        del self.model_path_
        del self.weights
        del self.output_layers_name
        del self.__new_model_path_
        del self.__quanti_layer_type_
        del self.__middle_layer_output_shape_
        del self.__merge_layer_type_
        del self.__merge_layer_indexs_
        del self.__merge_layer_shapes_


    def getGraph(self, onnx_path):
        model = onnx.load(onnx_path)
        model = shape_inference.infer_shapes(model)
        model_graph = model.graph
        graph = Graph.from_onnx(model_graph)
        graph = graph.transformed(transformers)
        graph.channel_dims = {}
        return graph

    def get_model(self):
        return self.model


    def get_all_output_layers_name(self):
        return self.middle_layers_name

    def get_input_layers_name(self):

        return self.input_layers_name

    def get_weight_data(self):
        return self.weights

    def get_merge_layer(self):
        merge_layers = OrderedDict()
        for i in self.__merge_layer_indexs_:
            merge_layers[i] = i
        return merge_layers

    def get_need_merge_layers(self):
        need_merge_layers = OrderedDict()
        for layer in self.__merge_layer_indexs_:
            need_merge_layers_ =  self.merge_layers_[layer]
            for need_layer in need_merge_layers_:
                need_merge_layers[need_layer] = layer

        return need_merge_layers

    def find_layer_index_by_name(self, layer_name, nodes):

        for i in range(len(nodes)):
            if layer_name == nodes[i].inputs[0]:
                return True, i

        return False, None

    def find_layer_index_by_current_name(self, layer_name, nodes):

        for i in range(len(nodes)):
            if layer_name == nodes[i].name:
                return True, i

        return False, None

    def __insert_output_node_to_model(self,
                                      model,
                                      save_model_path,
                                      insert_layer=None,
                                      middle_output_shapes=None,
                                      merge_layer_indexs=None,
                                      merge_layer_shapes=None):
        """
        向模型中插入输出节点、并将新的模型保存下来
        :param model: 模型
        :param insert_layer: 需要插入网络层名称
        :param save_model_path: 性模型的保存路径
        :return: 0
        """
        insert_layer_values = list(insert_layer.values())
        insert_layer_values.reverse()
        middle_output_shapes.reverse()
        for layer_info, shape in zip(insert_layer_values, middle_output_shapes):
            # key, value = layer_info

            if layer_info == '182':
                print('')
            prob_info = \
                helper.make_tensor_value_info(layer_info, onnx.TensorProto.FLOAT, shape)
            model.graph.output.insert(0, prob_info)

        if insert_layer and middle_output_shapes:
            merge_layer_indexs.reverse()
            merge_layer_shapes.reverse()
            for name, shape in zip(merge_layer_indexs, merge_layer_shapes):
                prob_info = \
                    helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
                model.graph.output.insert(0, prob_info)

        onnx.save(model, save_model_path)

        return 0

    def forword(self, x):
        """
        onnx模型的前向计算
        :param x:  模型的输入数据
        :return: 模型的输出数据
        """
        # x = x.transpose(0, 1, 3, 2)
        outputs = self.__sess_.run(self.output_layers_name[: 24], {self.input_layers_name[0]: x})
        if self.__merge_layer_indexs_:
            merger_layer_outputs = outputs[: len(self.__merge_layer_indexs_)]
            layer_outputs = outputs[len(self.__merge_layer_indexs_) : ]
        else:
            merger_layer_outputs = None
            layer_outputs = outputs
        return merger_layer_outputs, layer_outputs





import torch
class OnnxParserV2(object):
    __layer_type = ['BatchNormalization', 'MatMul']
    def __init__(self,
                 model_path):
        super().__init__()
        self.graph = self.getGraph(model_path)

        self.__param_ = {}
        pfe_key = 'voxel_feature_extractor'
        nodes = self.graph.nodes
        for node in nodes:
            if node.op_type == self.__layer_type[0]:
                input_tensors = node.input_tensors

                for key, v in input_tensors.items():
                    pfe_layer_key = key.lstrip(pfe_key).lstrip('.')
                    self.__param_[pfe_layer_key] = torch.tensor(v)

                del input_tensors, key, v

            elif node.op_type == self.__layer_type[1]:
                inputs = node.inputs
                Success, index = \
                    self.find_layer_index_by_current_name(inputs[1], nodes)
                if Success:
                    input_tensors = nodes[index].input_tensors
                    for key, v in input_tensors.items():
                        pfe_layer_key = key.lstrip(pfe_key).lstrip('.')
                        self.__param_[pfe_layer_key] = torch.tensor(v)

                    del input_tensors, key,  v
                del inputs, Success, index

            else:
                pass

        del pfe_key, nodes
        print('ok')
        pass

    def getGraph(self, onnx_path):
        model = onnx.load(onnx_path)
        model = shape_inference.infer_shapes(model)
        model_graph = model.graph
        graph = Graph.from_onnx(model_graph)
        graph = graph.transformed(transformers)
        graph.channel_dims = {}
        return graph

    def find_layer_index_by_current_name(self, layer_name, nodes):

        for i in range(len(nodes)):
            if layer_name == nodes[i].name:
                return True, i

        return False, None

    def get_param(self):
        return self.__param_
