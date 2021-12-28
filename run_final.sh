# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#         --model_name_or_path  voidful/albert_chinese_base \
#         --do_train --train_file CCPM-data/train.jsonl \
#         --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#         --test_file CCPM-data/split_test.jsonl \
#         --learning_rate 1e-5  --fp16 \
#         --num_train_epochs 2 \
#         --output_dir results/albert-base_lr1e-5_nte2_bsz16 \
#         --per_gpu_eval_batch_size=16 \
#         --per_device_train_batch_size=16 \
#         --overwrite_output

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#         --model_name_or_path  voidful/albert_chinese_large \
#         --do_train --train_file CCPM-data/train.jsonl \
#         --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#         --test_file CCPM-data/split_test.jsonl \
#         --learning_rate 1e-5  --fp16 \
#         --num_train_epochs 3 \
#         --output_dir results/albert-large_lr1e-5_nte3_bsz8 \
#         --per_gpu_eval_batch_size=16 \
#         --per_device_train_batch_size=8 \
#         --overwrite_output

#### RUN 1

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#           --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
#           --do_train --train_file CCPM-data/train.jsonl \
#           --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#           --test_file CCPM-data/split_test.jsonl \
#           --learning_rate 2e-5  --fp16 \
#           --num_train_epochs 1 \
#           --softmax_temperature 5 \
#           --evaluation_strategy epoch \
#           --output_dir results/ext-large_lr2e-5_tmpr5_nte1_bsz16 \
#           --per_gpu_eval_batch_size=16 \
#           --per_device_train_batch_size=16 \
#           --overwrite_output \
#         #   > ext-large.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -u main.py \
#           --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
#           --do_train --train_file CCPM-data/train.jsonl \
#           --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#           --test_file CCPM-data/split_test.jsonl \
#           --learning_rate 5e-6  --fp16 \
#           --num_train_epochs 3 \
#           --multiplier 3 \
#           --evaluation_strategy epoch \
#           --output_dir results/ext-large_lr5e-6_mtply3_nte3_bsz16 \
#           --per_gpu_eval_batch_size=16 \
#           --per_device_train_batch_size=16 \
#           --overwrite_output \
#         #   > ext-large.log 2>&1 &

#### RUN 2

CUDA_VISIBLE_DEVICES=2 python -u main.py \
          --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
          --do_train --train_file CCPM-data/train.jsonl \
          --do_eval  --validation_file CCPM-data/split_valid.jsonl \
          --test_file CCPM-data/split_test.jsonl \
          --learning_rate 5e-6  --fp16 \
          --num_train_epochs 2 \
          --softmax_temperature 3 \
          --evaluation_strategy epoch \
          --output_dir results/extlarge_postag_extra-freeze-randinit_lr5e-6_tmpr3_nte2_bsz16 \
          --per_gpu_eval_batch_size=16 \
          --per_device_train_batch_size=16 \
          --overwrite_output \
        #   > ext-large.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u main.py \
          --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
          --do_train --train_file CCPM-data/train.jsonl \
          --do_eval  --validation_file CCPM-data/split_valid.jsonl \
          --test_file CCPM-data/split_test.jsonl \
          --learning_rate 1e-5  --fp16 \
          --num_train_epochs 5 \
          --multiplier 5 \
          --evaluation_strategy epoch \
          --output_dir results/extlarge_postag_extra-freeze-randinit_lr1e-5_mtply5_nte5_bsz16 \
          --per_gpu_eval_batch_size=16 \
          --per_device_train_batch_size=16 \
          --overwrite_output \
        #   > ext-large.log 2>&1 &

#### RUN 3