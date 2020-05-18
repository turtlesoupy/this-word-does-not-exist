#!/bin/bash
PYTHON_PATH=/home/tdimson/anaconda3/envs/company_makeup/bin/python
LIBRARY_PATH=/home/tdimson/projects/company-makeup/title_maker_pro

export PYTHONPATH=/home/tdimson/projects/company-makeup: 

$PYTHON_PATH $LIBRARY_PATH/train.py \
--summary_comment=urban_dictionary_250_cleaned_top_defs_lr_00002_b9 \
--output_dir=models/urban_dictionary_250_cleaned_top_defs_lr_00002_b9 \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--train_data_file=/mnt/evo/projects/title-maker-pro/data/urban_dictionary_250_top_defs.pickle \
--do_eval \
--eval_data_file=/mnt/evo/projects/title-maker-pro/data/urban_dictionary_250_top_defs.pickle \
--per_gpu_train_batch_size 9 \
--per_gpu_eval_batch_size 9 \
--urban_dictionary_dataset \
--splits 0.95 --splits 0.05 \
--train_split_idx 0 --eval_split_idx 1 \
--title_scale 1.0 \
--evaluate_during_training \
--save_steps 10000 \
--logging_steps 5000 \
--eval_subsampling 1.0 \
--learning_rate 0.00002 \
--block_size 300 \
--num_train_epochs 20