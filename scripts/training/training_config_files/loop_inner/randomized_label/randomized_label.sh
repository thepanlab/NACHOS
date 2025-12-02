#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m training.training_sequential.loop_inner.training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/randomized_label/config_inner_InceptionV3.json

CUDA_VISIBLE_DEVICES=0 python -m training.training_sequential.loop_inner.training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/randomized_label/config_inner_InceptionV3_AILU486-left.json

CUDA_VISIBLE_DEVICES=1 python -m training.training_sequential.loop_inner.training_inner_loop --file /home/pcallec/mif_outer/scripts/training/training_config_files/loop_inner/randomized_label/config_inner_InceptionV3_AJBH331-left.json
