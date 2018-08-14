#!/bin/bash


# set the path of project root
proj_home=/home/djn/projects/DAC-Contest/DACSDC-DeepZ/Train

# set the path of training cfg
tr_cfg_path=${proj_home}/cfg/train1.cfg
# set the path of validation cfg
va_cfg_path=${proj_home}/cfg/valid1.cfg

# set the path of data root
data_home=${proj_home}/data
# set the path of training data
train_path=${data_home}/dataset/train_dataset.txt
# set the path of validation data
valid_path=${data_home}/dataset/valid_dataset.txt
# set the path of validation gt data
valid_gt_path=${data_home}/dataset/valid_dataset_label.txt

# set the name of the model
model_name=train1
# set the path to store trained models
model_path=${proj_home}/model
# set the path of the logging file of validation
log_path=${proj_home}/log
# set the path of the logging file of training
log_file=${proj_home}/log/${model_name}.out


# start training
# use -i param to choose the GPU device for training
# use -avg_loss param to set the value of initial loss
nohup \
	${proj_home}/darknet detector train \
	-i 1 \
	-tr_cfg ${tr_cfg_path} \
	-va_cfg ${va_cfg_path} \
	-tr_dir ${train_path} \
	-va_dir ${valid_path} \
	-va_gt_dir ${valid_gt_path} \
	-model_name ${model_name} \
	-model_dir ${model_path} \
	-log_dir ${log_path} \
  -avg_loss -1 > ${log_file} 2>&1 &