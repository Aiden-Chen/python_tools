from enum import Enum
''' FACE_DET onnnx model path '''
FACE_DET_MODEL_PATH = '../model/libfacedetection.onnx'

'''save new model path'''
NEW_FACE_DET_MODEL_PATH = '../save_model/libfacedetection_new.onnx'

'''save quanti ini path'''
FACE_DET_SAVE_QUANTI_PATH = '../ini/libfacedetection_quanti.ini'

'''quanti test data'''
QUANTI_DATA_PATH = '../data/'
IMG_SHAPE = [480, 640, 3]

'''量化位宽'''
FACE_DET_QUANTI_BITS = [8, 8]  # BITS[0]-->weight: intx   BITS[1]-->featuremap:uintx
FACE_DET_INPUT_QUANTI_BITS = 8

FACE_DET_INPUT_SHAPE = [1, 3, 480, 640]


"""   ***model common config***    """
'''输出量化方式是最大/小值 or kl divergence'''
'''navie: min&max   entropy: KL divergence'''
QUANTI_OUTPUT_METHOD = 'naive'

class WEIGHT_QUANTI_METHOD(Enum):
    CHANNEL_SYMMETRY = 0  # 逐通道对称量化
    LAYER_SYMMETRY = 1    # 逐层对称量化

WEIGHT_QUANTI_TYPE = WEIGHT_QUANTI_METHOD.CHANNEL_SYMMETRY

'''需要量化的量化层类型'''
QANTI_LAYER_TYPE = ['Relu']
MERGE_LAYER_TYPE = ['Concat']

log_cfg_path = "./log.cfg"

key_list_weight_quan_param = ["min_weight", "max_weight", "weight_scale", "weight_zero_point", "quan_weight_bits"]

key_list_input_quan_param = ["min_input", "max_input", "inputs_scale", "inputs_zero_point"]

key_list_output_quan_param = ["min_output", "max_output", "outputs_scale", "outputs_zero_point", "quan_in_bits",
                              "quan_out_bits"]


'''中间层的数据维度信息, intput: [1, 64, 320, 280]'''
FACE_DET_MIDDLE_LAYER_OUTPUT_SHAPE = [
    [1, 32, 240, 320],   # conv1 129
    [1, 16, 240, 320],   # conv2 132
    [1, 32, 120, 160],   # conv3 136
    [1, 32, 120, 160],   # conv4 139
    [1, 64, 60, 80],     # conv5 143
    [1, 32, 60, 80],     # conv6 146
    [1, 64, 60, 80],     # conv7 149
    [1, 128, 30, 40],    # conv8 153
    [1, 64, 30, 40],     # conv9 156
    [1, 128, 30, 40],    # conv10 159
    [1, 256, 15, 20],    # conv11 163
    [1, 128, 15, 20],    # conv12 166
    [1, 256, 15, 20],    # conv13 169
    [1, 256, 7, 10],     # conv14 173
    [1, 256, 7, 10],     # conv15 176
    [1, 256, 7, 10],     # conv16 179
    # output layer
    [1, 42, 60, 80],     # conv17 182
    [1, 6, 60, 80],      # conv18 184
    [1, 28, 30, 40],     # conv19 186
    [1, 4, 30, 40],      # conv20 188
    [1, 28, 15, 20],     # conv21 190
    [1, 4, 15, 20],      # conv22 192
    [1, 42, 7, 10],      # conv23 194
    [1, 6, 7, 10],       # conv24 196
]
FACE_DET_MERGE_LAYER_SHAPE_INDEX = []
FACE_DET_MERGE_LAYER_SHAPE = []

'''不需要量化的量化层名称'''
FACE_DET_DEQUANTI_LAYER_NAME = ['182', '184', '186', '188', '190', '192', '194', '196']
'''模型输入第一层'''
FIRST_LAYER = '129'
