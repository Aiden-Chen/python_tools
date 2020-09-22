#!/bin/bash


cd src
python3 pointpillars.py
exit

rm quantize.log -rf
rm data/* -rf
rm model/ -rf
mkdir model
cp -rf ../input/data/ ./ | tee quantize.log
cp -f ../input/saved_model/*.pt model/voxelnet.pt | tee -a quantize.log
cp -f ../input/saved_model/*.tckpt model/voxelnet.pt | tee -a quantize.log
cp -f ../2_model_transform/modify_second_ini/quantize_config.cfg src/config/quantize_config.cfg
cd ./src
echo `date +"%Y-%m-%d %H:%M:%S"` | tee -a ../quantize.log
python3 quantize_second.py | tee -a ../quantize.log
echo `date +"%Y-%m-%d %H:%M:%S"` | tee -a ../quantize.log
