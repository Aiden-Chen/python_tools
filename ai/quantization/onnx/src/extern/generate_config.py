import configparser
import argparse
import numpy as np
from extern.second.protos import pipeline_pb2
from google.protobuf import text_format

src_cfg_path = '../../cfg/second.cfg'
out_cfg_path = '../../cfg/second_new.cfg'
cfg_path = '../../model/pp_pretrain.config'


def main():
    parser = argparse.ArgumentParser(description="Demo of get_net_ini_from_four_cls_ini")
    parser.add_argument('-c', '--config', default=cfg_path)
    args = parser.parse_args()
    config_path = args.config
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    ranges = list(config.model.second.voxel_generator.point_cloud_range)
    voxel_size = list(config.model.second.voxel_generator.voxel_size)
    num_class = len(list(config.model.second.target_assigner.class_settings))
    rotations = list(config.model.second.target_assigner.class_settings[0].anchor_generator_range.rotations)
    print("## num_class is %d"% num_class)
    # if num_class != 4:
        # print("检测到模型为%d分类，仿真工具暂时只支持4分类" % (num_class,))
        # print("can't sponr %d" % (num_class))
        # exit(0)

    input_coor_dim = (np.array(ranges[3:]) - np.array(ranges[:3])) / np.array(voxel_size)
    output_coor_xy_dim = (input_coor_dim[:2]) / 8 + 0.5
    output_coor_xy_dim = output_coor_xy_dim.astype(np.int32)
    output_coor_xy_interval = output_coor_xy_dim - 1

    cls_ranges = []
    cls_wlhs = []
    for i in range(num_class):
        cls_range = list(config.model.second.target_assigner.class_settings[i].anchor_generator_range.anchor_ranges)
        cls_ranges.append(cls_range)
        sizes = list(config.model.second.target_assigner.class_settings[i].anchor_generator_range.sizes)
        cls_wlhs.append(sizes)

    sim_cfg = configparser.ConfigParser(delimiters=":")
    sim_cfg.read(src_cfg_path)

    anchors_x = ",".join(str(np.float32((cls_range[3] - cls_range[0]) / np.float32(output_coor_xy_interval[0]))) for cls_range in cls_ranges)
    anchors_y = ",".join(str(np.float32((cls_range[4] - cls_range[1]) / np.float32(output_coor_xy_interval[1]))) for cls_range in cls_ranges)

    anchors_z = ",".join(str(cls_range[2]) for cls_range in cls_ranges)
    anchors_w = ",".join(str(np.float32(cls_wlh[0])) for cls_wlh in cls_wlhs)
    anchors_l = ",".join(str(np.float32(cls_wlh[1])) for cls_wlh in cls_wlhs)
    anchors_h = ",".join(str(np.float32(cls_wlh[2])) for cls_wlh in cls_wlhs)
    anchors_r = ",".join(str(np.float32(r)) for r in rotations)
    anchors_x_min = ",".join(str(np.float32(cls_range[0])) for cls_range in cls_ranges)
    anchors_x_max = ",".join(str(np.float32(cls_range[3])) for cls_range in cls_ranges)
    anchors_y_min = ",".join(str(np.float32(cls_range[1])) for cls_range in cls_ranges)
    anchors_y_max = ",".join(str(np.float32(cls_range[4])) for cls_range in cls_ranges)
    anchors_z_min = ",".join(str(np.float32(cls_range[2])) for cls_range in cls_ranges)
    anchors_z_max = ",".join(str(np.float32(cls_range[5])) for cls_range in cls_ranges)
    sim_cfg['box_cls_para_cfg']['anchors_x'] = anchors_x
    sim_cfg['box_cls_para_cfg']['anchors_y'] = anchors_y
    sim_cfg['box_cls_para_cfg']['anchors_z'] = anchors_z
    sim_cfg['box_cls_para_cfg']['anchors_w'] = anchors_w
    sim_cfg['box_cls_para_cfg']['anchors_l'] = anchors_l
    sim_cfg['box_cls_para_cfg']['anchors_h'] = anchors_h
    sim_cfg['box_cls_para_cfg']['anchors_r'] = anchors_r
    sim_cfg['box_cls_para_cfg']['anchors_x_min'] = anchors_x_min
    sim_cfg['box_cls_para_cfg']['anchors_x_max'] = anchors_x_max
    sim_cfg['box_cls_para_cfg']['anchors_y_min'] = anchors_y_min
    sim_cfg['box_cls_para_cfg']['anchors_y_max'] = anchors_y_max
    sim_cfg['box_cls_para_cfg']['anchors_z_min'] = anchors_z_min
    sim_cfg['box_cls_para_cfg']['anchors_z_max'] = anchors_z_max
    sim_cfg['box_cls_para_cfg']['class_num'] = str(num_class)

    sim_cfg['vfe_func_cfg']['radar_data_x'] = ",".join((str(np.float32(ranges[0])), str(np.float32(ranges[3]))))
    sim_cfg['vfe_func_cfg']['radar_data_y'] = ",".join((str(ranges[1]), str(ranges[4])))
    sim_cfg['vfe_func_cfg']['radar_data_z'] = ",".join((str(ranges[2]), str(ranges[5])))
    # sim_cfg['vfe_func_cfg']['radar_data_r'] = ",".join((ranges[2], ranges[5]))
    sim_cfg['vfe_func_cfg']['valid_data_num'] = str(config.model.second.voxel_generator.max_number_of_points_per_voxel)
    sim_cfg['vfe_func_cfg']['voxel_size'] = ",".join(str(np.float32(i)) for i in voxel_size)

    with open(out_cfg_path, "w") as f:
        sim_cfg.write(f, space_around_delimiters=False)


if __name__ == '__main__':
    main()
