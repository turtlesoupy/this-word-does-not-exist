#!/bin/bash
PYTHON_PATH=/home/tdimson/anaconda3/envs/company_makeup/bin/python
LIBRARY_PATH=/home/tdimson/projects/company-makeup/title_maker_pro

$PYTHON_PATH $LIBRARY_PATH/train.py \
--summary_comment=en_dictionary_parsed_lr_00001_max_len_512_creativity_metrics \
--output_dir=models/en_dictionary_parsed_lr_00001_creativity \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--train_data_file=data/en_dictionary_parsed_randomized.pickle \
--do_eval \
--eval_data_file=data/en_dictionary_parsed_randomized.pickle \
--per_gpu_train_batch_size 6 \
--per_gpu_eval_batch_size 6 \
--gradient_accumulation_steps 1 \
--parsed_dictionary_dataset \
--splits 0.95 --splits 0.05 \
--train_split_idx 0 --eval_split_idx 1 \
--title_scale 1.0 \
--evaluate_during_training \
--save_steps 10000 \
--logging_steps 2500 \
--eval_subsampling 1.0 \
--learning_rate 0.00001 \
--block_size 512 \
--num_train_epochs 10 \
--eval_creativity_blacklist data/blacklist.pickle \
--num_eval_creativity 200 \
--eval_creativity_batch_size 50