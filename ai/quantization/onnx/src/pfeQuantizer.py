import config as cfg
from pfeparser.pfe import PillarFeatureNet, PointPillarsScatter
from pfeparser.pre_voxel import preprocess
import torch
from collections import OrderedDict
import pandas as pd
import heapq
import numpy as np

import logging.config
logging.config.fileConfig(cfg.log_cfg_path)

import configparser
config = configparser.ConfigParser(delimiters=":")

from onnxparser.onnx_parser import OnnxParserV2
from quantize_base.quantize_base_kl_v2 import ThresholdLayerOutputs

class PFEQuantizer(object):
    __max_input_ = OrderedDict()
    __min_input_ = OrderedDict()
    __max_output_ = OrderedDict()
    __min_output_ = OrderedDict()
    quanti_layer_name = ['164']

    def __init__(self,
                 voxel_size,
                 point_clound_range,
                 scatter_output_shape,
                 max_points_per_voxel,
                 input_quanti_bits,
                 output_quanti_method,
                 quanti_bits,
                 ini_file):
        super().__init__()
        self.name_ = 'PFEQuantizer'
        # self.pfe_net_ = PillarFeatureNet(voxel_size=voxel_size, pc_range=point_clound_range)
        self.pfe_net_ = PillarFeatureNet(voxel_size=voxel_size, pc_range=point_clound_range)
        self.pfe_model()
        self.scatter_ = PointPillarsScatter(output_shape=scatter_output_shape)
        self.voxel_size_ = voxel_size
        self.point_clound_range_ = point_clound_range
        self.max_points_per_voxel_ = max_points_per_voxel
        self.input_quanti_bits_ = input_quanti_bits
        self.output_quanti_method_ = output_quanti_method
        self.quanti_bits_ = quanti_bits
        self.ini_file_ = ini_file

    def __del__(self):
        del self.name_
        del self.pfe_net_
        del self.scatter_
        del self.voxel_size_
        del self.point_clound_range_
        del self.max_points_per_voxel_
        del self.input_quanti_bits_
        del self.output_quanti_method_
        del self.quanti_bits_
        del self.ini_file_

    def pfe_model(self):
        # pfe_net_dict = self.pfe_net_.state_dict()
        # model = torch.load(cfg.PFE_MODEL_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu")
        #
        # pfe_key = 'voxel_feature_extractor'
        # model_pfe_param = {}
        # for k, v in model.items():
        #     pfe_layer_key = k.lstrip(pfe_key).lstrip('.')
        #     if pfe_layer_key in pfe_net_dict:
        #         model_pfe_param[pfe_layer_key] = v
        #     else:
        #         pass
        #
        # self.__weights_ = model_pfe_param['pfn_layers.0.linear.weight'].detach().numpy()
        #
        # pfe_net_dict.update(model_pfe_param)
        # self.pfe_net_.load_state_dict(pfe_net_dict)
        # self.pfe_net_.eval()

        pfe_parser = OnnxParserV2(cfg.PFE_MODEL_PATH)
        pfe_net_dict_ = self.pfe_net_.state_dict()
        # model_ = torch.load(cfg.PFE_MODEL_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu")
        params = pfe_parser.get_param()
        self.__weights_ = params['pfn_layers.0.linear.weight'].detach().numpy()
        pfe_net_dict_.update(params)
        self.pfe_net_.load_state_dict(pfe_net_dict_)
        self.pfe_net_.eval()


        print('fpe model success')

    def forword(self, x):
        print('*********************file_name: {}**********************'.format(x))
        example = preprocess(x, self.voxel_size_, self.point_clound_range_, self.max_points_per_voxel_)
        features = torch.from_numpy(example['voxels'])
        num_voxels = torch.from_numpy(example['num_points'])
        coors = torch.from_numpy(example['coordinates'])

        pfe_output = self.pfe_net_(features, num_voxels, coors)

        # pfe_output = self.pfe_net_(features, num_voxels, coors)
        scatter_output = self.scatter_(pfe_output, coors, 1)

        self.get_layer_min_max(OrderedDict([(self.quanti_layer_name[0], self.quanti_layer_name[0]), ]),
                               [scatter_output.detach().numpy()])
        self.get_input_min_max(self.quanti_layer_name, features.detach().numpy())

        del example, features, num_voxels, coors

        return scatter_output

    def save_param(self):
        parm_list = []
        val_list = []

        # 权重量化
        weight_scale, weight_zero_point, weight_max, weight_min = \
            self.quanti_weight_channel_symmetry_quantitative(self.__weights_, self.quanti_bits_[0])

        # 输入量化
        input_quanti_bits_ = self.input_quanti_bits_
        min_val_input, max_val_input, input_scale_float, input_zero_point_float = \
            self.get_quanti_param(self.__min_input_[self.quanti_layer_name[0]],
                                  self.__max_input_[self.quanti_layer_name[0]],
                                  input_quanti_bits_)
        parm_list.extend(cfg.key_list_input_quan_param)
        val_list.extend([min_val_input, max_val_input, input_scale_float, input_zero_point_float])

        # 输出量化
        min_val, max_val, outputs_scale_float, outputs_zero_point_float = \
            self.get_quanti_param(self.__min_output_[self.quanti_layer_name[0]],
                                  self.__max_output_[self.quanti_layer_name[0]],
                                  self.quanti_bits_[1])
        parm_list.extend(cfg.key_list_weight_quan_param + cfg.key_list_output_quan_param)
        val_list.extend([weight_min,
                         weight_max,
                         str(weight_scale).strip('[]'),
                         str(weight_zero_point).strip('[]'),
                         self.quanti_bits_[0],
                         min_val,
                         max_val,
                         outputs_scale_float,
                         outputs_zero_point_float,
                         input_quanti_bits_,
                         self.quanti_bits_[1]])

        self.write_ini_file(self.ini_file_, self.quanti_layer_name[0], parm_list, val_list)


        pass

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
            if self.output_quanti_method_ == 'naive':
                # min&max quanti
                max_v = np.max(value)
                min_v = np.min(value)
            elif self.output_quanti_method_ == 'entropy':
                # KL divergence quanti
                min_v, max_v = ThresholdLayerOutputs(value, self.quanti_bits_[1])
            else:
                raise ValueError(
                    'unknown quantize mode %s received, expected `naive`, or `entropy`' % self.output_quanti_method_)

            # store max&min value
            self.__min_output_[key] = min_v \
                if key not in self.__min_output_ else np.hstack([self.__min_output_[key], min_v])
            self.__max_output_[key] = max_v \
                if key not in self.__max_output_ else np.hstack([self.__max_output_[key], max_v])

        return 0

    def get_input_min_max(self, input_layer_name, data):
        """
        对模型的每一层（需要量化的网络层）的输出结果获取其中的最大最小值，
        获取最大/小值的方法有：KL divergence and Max&Min
        :param output_data: 需要量化的网络层的输出结果
        :return: 0
        """
        if self.output_quanti_method_ == 'naive':
            # min&max quanti
            max_v = np.max(data)
            min_v = np.min(data)
        elif self.output_quanti_method_ == 'entropy':
            # KL divergence quanti
            min_v, max_v = ThresholdLayerOutputs(data, self.input_quanti_bits_)
        else:
            raise ValueError(
                'unknown quantize mode %s received, expected `naive`, or `entropy`' % self.output_quanti_method_)

        # store max&min value
        self.__min_input_[input_layer_name[0]] = min_v \
            if input_layer_name[0] not in self.__min_input_ \
            else np.hstack([self.__min_input_[input_layer_name[0]], min_v])

        self.__max_input_[input_layer_name[0]] = max_v \
            if input_layer_name[0] not in self.__max_input_ \
            else np.hstack([self.__max_input_[input_layer_name[0]], max_v])

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

