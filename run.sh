#!/bin/bash

# for lr in 2e-5 5e-5 1e-4; do
#     for nte in 2 3 4; do
#         for bsz in 8 16 24; do
# CUDA_VISIBLE_DEVICES=2 python -u main.py \
#           --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
#           --do_train --train_file CCPM-data/train.jsonl \
#           --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#           --test_file CCPM-data/split_test.jsonl \
#           --learning_rate ${lr}  --fp16 \
#           --num_train_epochs ${nte} \
#           --output_dir results/ext-large_lr${lr}_nte${nte}_bsz${bsz} \
#           --per_gpu_eval_batch_size=16 \
#           --per_device_train_batch_size=${bsz} \
#           --overwrite_output \
#         #   > ext-large.log 2>&1 &
#         done
#     done
# done

# TUNE 2
# models=("voidful/albert_chinese_tiny" "hfl/chinese-macbert-base" "hfl/chinese-macbert-large")

# for modelname in ${models[@]}; do
#     for lr in 2e-5 5e-5 1e-4; do
#         for nte in 2 3 4; do
#             for bsz in 8 16 24; do
#     CUDA_VISIBLE_DEVICES=2 python -u main.py \
#             --model_name_or_path  ${modelname} \
#             --do_train --train_file CCPM-data/train.jsonl \
#             --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#             --test_file CCPM-data/split_test.jsonl \
#             --learning_rate ${lr}  --fp16 \
#             --num_train_epochs ${nte} \
#             --output_dir results/${modelname}_lr${lr}_nte${nte}_bsz${bsz} \
#             --per_gpu_eval_batch_size=16 \
#             --per_device_train_batch_size=${bsz} \
#             --overwrite_output \
#             #   > ext-large.log 2>&1 &
#             done
#         done
#     done
# done

# TUNE 3
# models=("hfl/chinese-macbert-large")

# for modelname in ${models[@]}; do
#     for lr in 2e-5 5e-5 1e-4; do
#         for nte in 2 3 4; do
#             for bsz in 8 16 24; do
#     CUDA_VISIBLE_DEVICES=2 python -u main.py \
#             --model_name_or_path  ${modelname} \
#             --do_train --train_file CCPM-data/train.jsonl \
#             --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#             --test_file CCPM-data/split_test.jsonl \
#             --learning_rate ${lr}  --fp16 \
#             --num_train_epochs ${nte} \
#             --output_dir results/${modelname}_lr${lr}_nte${nte}_bsz${bsz} \
#             --per_gpu_eval_batch_size=16 \
#             --per_device_train_batch_size=${bsz} \
#             --overwrite_output \
#             #   > ext-large.log 2>&1 &
#             done
#         done
#     done
# done

# TUNE 4
# for lr in 2e-5 5e-5 1e-4; do
#     for nte in 2 3 4; do
#         for bsz in 8 16; do
# CUDA_VISIBLE_DEVICES=2 python -u main.py \
#         --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
#         --do_train --train_file CCPM-data/train.jsonl \
#         --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#         --test_file CCPM-data/split_test.jsonl \
#         --learning_rate ${lr}  --fp16 \
#         --num_train_epochs ${nte} \
#         --output_dir results/extlarge_postag_extra-freeze_lr${lr}_nte${nte}_bsz${bsz} \
#         --per_gpu_eval_batch_size=16 \
#         --per_device_train_batch_size=${bsz} \
#         --overwrite_output \
#         #   > ext-large.log 2>&1 &
#         done
#     done
# done

# TUNE 5
# for nte in 2 3 4; do
#     CUDA_VISIBLE_DEVICES=2 python -u main.py \
#             --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
#             --do_train --train_file CCPM-data/train.jsonl \
#             --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#             --test_file CCPM-data/split_test.jsonl \
#             --learning_rate 5e-5  --fp16 \
#             --num_train_epochs ${nte} \
#             --output_dir results/extlarge_postag_extra-freeze_lr5e-5_nte${nte}_bsz16_factorNone \
#             --per_gpu_eval_batch_size=16 \
#             --per_device_train_batch_size=16 \
#             --overwrite_output
# done

# for nte in 2 3 4; do
#     for factor in 0.1 0.5 1 5 10; do
#     CUDA_VISIBLE_DEVICES=2 python -u main.py \
#             --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
#             --do_train --train_file CCPM-data/train.jsonl \
#             --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#             --test_file CCPM-data/split_test.jsonl \
#             --learning_rate 5e-5  --fp16 \
#             --num_train_epochs ${nte} \
#             --pos_info_factor ${factor} \
#             --output_dir results/extlarge_postag_extra-freeze_lr5e-5_nte${nte}_bsz16_factor${factor} \
#             --per_gpu_eval_batch_size=16 \
#             --per_device_train_batch_size=16 \
#             --overwrite_output \
#             #   > ext-large.log 2>&1 &
#     done
# done

# TUNE 6
# for nte in 2 3 4; do
#     CUDA_VISIBLE_DEVICES=0 python -u main.py \
#             --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
#             --do_train --train_file CCPM-data/train.jsonl \
#             --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#             --test_file CCPM-data/split_test.jsonl \
#             --learning_rate 2e-5  --fp16 \
#             --num_train_epochs ${nte} \
#             --output_dir results/extlarge_postag_extra-freeze_lr2e-5_nte${nte}_bsz8_factorNone \
#             --per_gpu_eval_batch_size=16 \
#             --per_device_train_batch_size=8 \
#             --overwrite_output
# done

# for nte in 2 3 4; do
#     for factor in 0.1 0.5 1 5 10; do
#     CUDA_VISIBLE_DEVICES=0 python -u main.py \
#             --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
#             --do_train --train_file CCPM-data/train.jsonl \
#             --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#             --test_file CCPM-data/split_test.jsonl \
#             --learning_rate 2e-5  --fp16 \
#             --num_train_epochs ${nte} \
#             --pos_info_factor ${factor} \
#             --output_dir results/extlarge_postag_extra-freeze_lr2e-5_nte${nte}_bsz8_factor${factor} \
#             --per_gpu_eval_batch_size=16 \
#             --per_device_train_batch_size=8 \
#             --overwrite_output \
#             #   > ext-large.log 2>&1 &
#     done
# done

# TUNE 7
# for nte in 3 4 5 6; do
#     for lr in 2e-5 5e-5 1e-4 2e-4; do
# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#           --model_name_or_path  google/mt5-small \
#           --do_train --train_file CCPM-data/train.jsonl \
#           --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#           --test_file CCPM-data/split_test.jsonl \
#           --learning_rate ${lr}  --fp16 \
#           --num_train_epochs ${nte} \
#           --output_dir results_mt5_lr${lr}_nte${nte} \
#           --per_gpu_eval_batch_size=16 \
#           --per_device_train_batch_size=16 \
#           --overwrite_output
#     done
# done

# TUNE 8
for nte in 2 3 4 5 6; do
    for lr in 1e-5 5e-6 2e-6; do
CUDA_VISIBLE_DEVICES=0 python -u main.py \
          --model_name_or_path  google/mt5-small \
          --do_train --train_file CCPM-data/train.jsonl \
          --do_eval  --validation_file CCPM-data/split_valid.jsonl \
          --test_file CCPM-data/split_test.jsonl \
          --learning_rate ${lr}  --fp16 \
          --num_train_epochs ${nte} \
          --output_dir results_mt5_lr${lr}_nte${nte} \
          --per_gpu_eval_batch_size=16 \
          --per_device_train_batch_size=16 \
          --overwrite_output
    done
done