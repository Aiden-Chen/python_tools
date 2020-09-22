
import numpy as np
import pandas as pd
import heapq

import configparser
from collections import OrderedDict
import logging
import logging.config

from onnxparser.onnx_parser import OnnxPareser
from quantize_base.quantize_base_kl_v2 import ThresholdLayerOutputs
import config as cfg

logging.config.fileConfig(cfg.log_cfg_path)
config = configparser.ConfigParser(delimiters=":")

class Quantizer(object):
    __max_output_data_ = OrderedDict()
    __min_output_data_ = OrderedDict()
    __max_input_ = OrderedDict()
    __min_input_ = OrderedDict()
    __weight_scale_ = OrderedDict()
    __weight_zero_point_ = OrderedDict()
    __weight_max_ = OrderedDict()
    __weight_min_ = OrderedDict()
    __weight_quanti_bits_ = OrderedDict()
    __merge_layer_max_ = OrderedDict()
    __merge_layer_min_ = OrderedDict()
    __detect_thresh_ = 0

    def __init__(self, model_path,
                 ini_file=None,
                 input_quanti_bits=None,
                 quanti_bits=None,
                 output_quanti_method=None,
                 weight_quanti_type=None,
                 save_new_model_path=None,
                 quanti_layer_type=None,
                 middle_layer_output_shape=None,
                 merge_layer_type=None,
                 merge_layer_indexs=None,
                 merge_layer_shapes=None,
                 dequan_layer_name=None,
                 do_detection=False,
                 input_do_quanti=False):
        super().__init__()
        self.__parser_ = OnnxPareser(model_path,
                                     save_new_model_path,
                                     quanti_layer_type,
                                     middle_layer_output_shape,
                                     merge_layer_type,
                                     merge_layer_indexs,
                                     merge_layer_shapes)
        self.__ini_file_ = ini_file
        self.__input_quanti_bits_ = input_quanti_bits
        self.__quantize_bitwise_ = quanti_bits
        self.__output_quanti_method_ = output_quanti_method
        self.__quanti_layer_ = self.__parser_.get_all_output_layers_name()
        self.__dequan_layer_name_ = dequan_layer_name
        self.__do_detection_ = do_detection
        self.__input_do_quanti_ = input_do_quanti
        self.__weight_quanti_type_ = weight_quanti_type

    def __del__(self):
        del self.__parser_
        del self.__ini_file_
        del self.__quantize_bitwise_
        del self.__output_quanti_method_
        del self.__quanti_layer_
        del self.__dequan_layer_name_
        del self.__do_detection_
        del self.__input_do_quanti_
        del self.__input_quanti_bits_
        del self.__weight_quanti_type_

    def quanti_weight_channel_symmetry_quantitative(self, weights, quanti_bits):
        """
        量化权重数据、权重的数据shape为k,c,h,w
        量化方式为逐通道、对称量化
        :param weights: 需要量化的权重
        :param quanti_bis: 权重量化的位宽
        :return: 返回权重的量化系数、量化零点、最大值、最小值、
        """
        weight_scale = []
        weight_zero_point = []
        max_value = pow(2, quanti_bits - 1) - 1

        for i in range(len(weights)):
            weights_k = weights[i]
            weight_k_max = np.max(np.abs(weights_k))
            weight_scale_float = weight_k_max / max_value
            weight_scale.append(weight_scale_float)
            weight_zero_point.append(0)
            del weights_k, weight_k_max, weight_scale_float

        del max_value
        return weight_scale, weight_zero_point, np.max(weights), np.min(weights)

    def quanti_weight_layer_symmetry_quantitative(self, weights, quanti_bits):
        """
        量化权重数据、权重的数据shape为k,c,h,w
        量化方式为逐通道、对称量化
        :param weights: 需要量化的权重
        :param quanti_bis: 权重量化的位宽
        :return: 返回权重的量化系数、量化零点、最大值、最小值、
        """
        weight_scale = []
        weight_zero_point = []
        max_value = pow(2, quanti_bits - 1) - 1
        max_k = np.max(np.abs(weights))

        for i in range(len(weights)):
            weight_scale_float = max_k / max_value
            weight_scale.append(weight_scale_float)
            weight_zero_point.append(0)
            del weight_scale_float

        del max_value, max_k
        return weight_scale, weight_zero_point, np.max(weights), np.min(weights)

    def forword(self, x=None):
        """
        量化的前向，用于获取中间数据
        :param input_data_path: 输入数据路径
        :return: 返回正整网络的数据结果
        """

        merge_layer_outputs, layer_outputs = self.__parser_.forword(x)

        merge_layers = self.__parser_.get_merge_layer()
        if len(merge_layers) != 0:
            self.get_layer_min_max(merge_layers, merge_layer_outputs)
        self.get_layer_min_max(self.__quanti_layer_, layer_outputs)
        self.get_input_min_max(self.__parser_.input_layers_name, x)

        # 保存Detection量化信息
        if self.__do_detection_:
            layers_list = list(self.__quanti_layer_.values())
            output_data = layer_outputs[len(layers_list) - 3: len(layers_list)]
            detection_out = \
                self.__detection_.generate_bbox(output_data[0], output_data[1],
                                                output_data[2])
            if detection_out is not None and self.__do_detection_:
                max_val_detect = np.max(np.abs(detection_out))
                logging.debug(
                    "detection_max:{}, detection_min:{},".format(np.max(detection_out), np.min(detection_out)))
                # detect_thresh_list.append(max_val_detect)
                if max_val_detect >= self.__detect_thresh_ and max_val_detect is not None:
                    self.__detect_thresh_ = max_val_detect
                del max_val_detect
            del output_data, detection_out

        del merge_layer_outputs,  merge_layers

        return layer_outputs[len(list(self.__quanti_layer_.values())) - 3: len(list(self.__quanti_layer_.values()))]


    def save_param(self):
        """
        保存量化系数、量化零点、量化位宽、最大最小值
        :return: 0
        """
        # 量化权重
        weights = self.__parser_.get_weight_data()
        for key, value in weights.items():
            if self.__weight_quanti_type_ == cfg.WEIGHT_QUANTI_METHOD.CHANNEL_SYMMETRY:
                weight_scale, weight_zero_point, weight_max, weight_min = \
                    self.quanti_weight_channel_symmetry_quantitative(value, self.__quantize_bitwise_[0])
            elif self.__weight_quanti_type_ == cfg.WEIGHT_QUANTI_METHOD.CHANNEL_SYMMETRY:
                weight_scale, weight_zero_point, weight_max, weight_min = \
                    self.quanti_weight_layer_symmetry_quantitative(value, self.__quantize_bitwise_[0])
            else:
                raise ValueError("don't support this weight quanti type {}".format(self.__weight_quanti_type_))

            self.__weight_scale_[key] = weight_scale
            self.__weight_zero_point_[key] = weight_zero_point
            self.__weight_max_[key] = weight_max
            self.__weight_min_[key] = weight_min
            self.__weight_quanti_bits_[key] = self.__quantize_bitwise_[0]
            del weight_scale, weight_zero_point, weight_max, weight_min
        del weights

        weight_layer = list(self.__weight_min_.keys())
        # 合并量化参数
        need_merge_layers = self.__parser_.get_need_merge_layers()
        # 量化输入输出
        for key, value in self.__max_output_data_.items():
            logging.debug(key)
            parm_list = []
            val_list = []

            quan_out_bits = self.__quantize_bitwise_[1]

            input_quanti_bits_ = quan_out_bits
            if key == cfg.FIRST_LAYER:
                #计算第一层输入量化尺度
                input_quanti_bits_ = self.__input_quanti_bits_
                input_name = self.__parser_.get_input_layers_name()
                min_val_input, max_val_input, input_scale_float, input_zero_point_float =\
                    self.get_quanti_param(self.__min_input_[input_name[0]],
                                          self.__max_input_[input_name[0]],
                                          input_quanti_bits_)

                if not self.__input_do_quanti_:
                    input_scale_float = 1
                    input_zero_point_float = 0

                parm_list.extend(cfg.key_list_input_quan_param)
                val_list.extend([min_val_input, max_val_input, input_scale_float, input_zero_point_float])
                del input_name, \
                    min_val_input, max_val_input, input_scale_float, input_zero_point_float

            # 计算输出层量化尺度
            if key in weight_layer and key not in list(need_merge_layers.keys()):
                # 计算卷积层的量化参数，且输出量化参数是不需要合并的
                min_val, max_val, outputs_scale_float, outputs_zero_point_float =\
                    self.get_quanti_param(self.__min_output_data_[key],
                                          self.__max_output_data_[key],
                                          quan_out_bits)
                parm_list.extend(cfg.key_list_weight_quan_param + cfg.key_list_output_quan_param)
                val_list.extend([self.__weight_min_[key],
                                 self.__weight_max_[key],
                                 str(self.__weight_scale_[key]).strip('[]'),
                                 str(self.__weight_zero_point_[key]).strip('[]'),
                                 self.__weight_quanti_bits_[key],
                                 min_val,
                                 max_val,
                                 outputs_scale_float,
                                 outputs_zero_point_float,
                                 input_quanti_bits_,
                                 quan_out_bits])
                del max_val, min_val, input_quanti_bits_, outputs_zero_point_float, outputs_scale_float
            elif key in weight_layer and key in list(need_merge_layers.keys()):
                # 计算卷积层的量化参数，且输出量化参数是需要合并的
                min_val, max_val, outputs_scale_float, outputs_zero_point_float =\
                    self.get_quanti_param(self.__min_output_data_[need_merge_layers[key]],
                                          self.__max_output_data_[need_merge_layers[key]],
                                          quan_out_bits)
                parm_list.extend(cfg.key_list_weight_quan_param + cfg.key_list_output_quan_param)
                val_list.extend([self.__weight_min_[key],
                                 self.__weight_max_[key],
                                 str(self.__weight_scale_[key]).strip('[]'),
                                 str(self.__weight_zero_point_[key]).strip('[]'),
                                 self.__weight_quanti_bits_[key],
                                 min_val,
                                 max_val,
                                 outputs_scale_float,
                                 outputs_zero_point_float,
                                 input_quanti_bits_,
                                 quan_out_bits])
                del max_val, min_val, input_quanti_bits_, outputs_zero_point_float, outputs_scale_float
            else:
                # 计算非卷积层的量化参数
                min_val, max_val, outputs_scale_float, outputs_zero_point_float =\
                    self.get_quanti_param(self.__min_output_data_[key],
                                          self.__max_output_data_[key],
                                          quan_out_bits)
                parm_list.extend(cfg.key_list_output_quan_param)
                val_list.extend([min_val,
                                 max_val,
                                 outputs_scale_float,
                                 outputs_zero_point_float,
                                 input_quanti_bits_,
                                 quan_out_bits])
                del max_val, min_val, input_quanti_bits_, outputs_zero_point_float, outputs_scale_float

            if key in self.__dequan_layer_name_:
                parm_list.append('output_dequan')
                val_list.append('false')

            self.write_ini_file(self.__ini_file_, key, parm_list, val_list)
            del parm_list, val_list, key

        del need_merge_layers, weight_layer


        # detection quantization
        if self.__do_detection_:
            quan_detect_bits = 16
            detection_thresh = self.__detect_thresh_
            detection_scale = detection_thresh / (pow(2, quan_detect_bits - 1) - 1)
            parm_list = cfg.key_list_output_quan_param[:-1]
            val_list = [-detection_thresh, detection_thresh, detection_scale, 0, quan_detect_bits]
            self.write_ini_file(self.__ini_file_, 'Detection_corner_box2d', parm_list, val_list)
            del quan_detect_bits, detection_thresh, detection_scale, parm_list, val_list

        return 0


    def write_ini_file(self, ini_file, layer_name, parm_list, val_list):
        """
        Write the quantization parameters to a file with the suffix ini
        :param layer_name: layer name to be write
        :param parm_list: a list of parameters name
        :param val_list: a list of values corresponding to the parameters name order in parm_list
        :return:
        """
        dict_parm_val = OrderedDict()
        for k, v in zip(parm_list, val_list):
            print(k)
            dict_parm_val[k] = v
        config[layer_name] = dict_parm_val
        with open(ini_file, 'w') as configfile:
            config.write(configfile, space_around_delimiters=False)

        del dict_parm_val

    def get_quanti_param(self, min_list_all, max_list_all, quanti_bits, n=100):
        min_list = pd.Series(min_list_all)
        data_count = min_list.value_counts(bins=2, normalize=True, sort=False)
        print(data_count)
        for k, v in data_count.items():
            if v > 0.7:
                num_thresh = str(k).split(',')[-1].split(']')[0]  # k.right即可代替，取区间的右值
                min_list = np.array(min_list[min_list.astype(np.float32) <= float(num_thresh)])
        if len(min_list) > n:
            min_list = np.array(min_list)
            min_list_index = heapq.nsmallest(n, range(len(min_list)), min_list.take)
            min_list = min_list[min_list_index]
        elif len(min_list) == 0:
            min_list = min_list_all
        min_input_all = np.mean(min_list)

        max_list = pd.Series(max_list_all)
        data_count = max_list.value_counts(bins=2, normalize=True, sort=False)
        print(data_count)
        for k, v in data_count.items():
            if v > 0.7:
                num_thresh = str(k).split(',')[0].split('(')[1]  # k.left代替即可
                max_list = np.array(max_list[max_list.astype(np.float32) >= float(num_thresh)])  # 理论上应该取大于
        if len(max_list) > n:
            max_list = np.array(max_list)
            max_list_index = heapq.nlargest(n, range(len(max_list)), max_list.take)
            max_list = max_list[max_list_index]
        elif len(max_list) == 0:
            max_list = max_list_all
        max_input_all = np.mean(max_list)

        outputs_scale_float = (max_input_all - min_input_all) / (pow(2, quanti_bits) - 1)
        outputs_zero_point_float = -1 * min_input_all / outputs_scale_float
        return min_input_all, max_input_all, outputs_scale_float, outputs_zero_point_float

    def get_layer_min_max(self, quanti_layers, output_data):
        """
        对模型的每一层（需要量化的网络层）的输出结果获取其中的最大最小值，
        获取最大/小值的方法有：KL divergence and Max&Min
        :param output_data: 需要量化的网络层的输出结果
        :return: 0
        """
        it = iter(output_data)
        for key, relu_name in quanti_layers.items():
            value = next(it)
            if self.__output_quanti_method_ == 'naive':
                # min&max quanti
                max_v = np.max(value)
                min_v = np.min(value)
            elif self.__output_quanti_method_ == 'entropy':
                # KL divergence quanti
                min_v, max_v = ThresholdLayerOutputs(value, self.__quantize_bitwise_[1])
            else:
                raise ValueError(
                    'unknown quantize mode %s received, expected `naive`, or `entropy`' % self.__output_quanti_method_)

            # store max&min value
            self.__min_output_data_[key] = min_v \
                if key not in self.__min_output_data_ else np.hstack([self.__min_output_data_[key], min_v])
            self.__max_output_data_[key] = max_v \
                if key not in self.__max_output_data_ else np.hstack([self.__max_output_data_[key], max_v])

        return 0

    def get_input_min_max(self, input_layer_name, data):
        """
        对模型的每一层（需要量化的网络层）的输出结果获取其中的最大最小值，
        获取最大/小值的方法有：KL divergence and Max&Min
        :param output_data: 需要量化的网络层的输出结果
        :return: 0
        """
        if self.__output_quanti_method_ == 'naive':
            # min&max quanti
            max_v = np.max(data)
            min_v = np.min(data)
        elif self.__output_quanti_method_ == 'entropy':
            # KL divergence quanti
            min_v, max_v = ThresholdLayerOutputs(data, self.__quantize_bitwise_[1])
        else:
            raise ValueError(
                'unknown quantize mode %s received, expected `naive`, or `entropy`' % self.__output_quanti_method_)

        # store max&min value
        self.__min_input_[input_layer_name[0]] = min_v \
            if input_layer_name[0] not in self.__min_input_ \
            else np.hstack([self.__min_input_[input_layer_name[0]], min_v])

        self.__max_input_[input_layer_name[0]] = max_v \
            if input_layer_name[0] not in self.__max_input_ \
            else np.hstack([self.__max_input_[input_layer_name[0]], max_v])


