
from onnxquantizer import Quantizer
import config as cfg
import os
import cv2
import numpy as np


def prehandle(img_path, dst_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized = cv2.resize(img, dsize=(dst_size[1], dst_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized


def main():

    #load rpn model
    rpn_model = Quantizer(model_path=cfg.FACE_DET_MODEL_PATH,
                          ini_file=cfg.FACE_DET_SAVE_QUANTI_PATH,
                          input_quanti_bits=cfg.FACE_DET_INPUT_QUANTI_BITS,
                          quanti_bits=cfg.FACE_DET_QUANTI_BITS,
                          output_quanti_method=cfg.QUANTI_OUTPUT_METHOD,
                          weight_quanti_type=cfg.WEIGHT_QUANTI_TYPE,
                          save_new_model_path=cfg.NEW_FACE_DET_MODEL_PATH,
                          quanti_layer_type=cfg.QANTI_LAYER_TYPE,
                          middle_layer_output_shape=cfg.FACE_DET_MIDDLE_LAYER_OUTPUT_SHAPE,
                          merge_layer_type=cfg.MERGE_LAYER_TYPE,
                          merge_layer_indexs=cfg.FACE_DET_MERGE_LAYER_SHAPE_INDEX,
                          merge_layer_shapes=cfg.FACE_DET_MERGE_LAYER_SHAPE,
                          dequan_layer_name=cfg.FACE_DET_DEQUANTI_LAYER_NAME,
                          do_detection=False,
                          input_do_quanti=False)

    file_list = os.listdir(cfg.QUANTI_DATA_PATH)

    for file_name in file_list:
        file_name = cfg.QUANTI_DATA_PATH + file_name
        input_data = prehandle(file_name, cfg.IMG_SHAPE)
        input_data = input_data.transpose(2, 0, 1)
        input_data_ = input_data.flatten().reshape(cfg.FACE_DET_INPUT_SHAPE)   # = pfe_output.detach().numpy()
        rpn_model.forword(np.array(input_data_.astype(np.float32)))

        print('*********************************************')
        # break

    print('save param...')
    rpn_model.save_param()

    return


if __name__ == '__main__':
    main()
