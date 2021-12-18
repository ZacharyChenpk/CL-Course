#!/usr/bin/env bash
python -u main.py \
          --model_name_or_path  hfl/chinese-roberta-wwm-ext \
          --do_train --train_file CCPM-data/train.jsonl \
          --do_eval  --validation_file CCPM-data/split_valid.jsonl \
          --test_file CCPM-data/split_test.jsonl \
          --learning_rate 5e-5  --fp16 \
          --num_train_epochs 3 \
          --output_dir results_with_tag \
          --per_gpu_eval_batch_size=16 \
          --per_device_train_batch_size=16 \
          --overwrite_output