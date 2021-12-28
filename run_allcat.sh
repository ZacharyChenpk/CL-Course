for lr in 5e-6 1e-5 2e-5; do
CUDA_VISIBLE_DEVICES=2 python -u main_allcat.py \
        --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
        --do_train --train_file CCPM-data/train.jsonl \
        --do_eval  --validation_file CCPM-data/split_valid.jsonl \
        --test_file CCPM-data/split_test.jsonl \
        --learning_rate ${lr}  --fp16 \
        --num_train_epochs 5 \
        --evaluation_strategy epoch \
        --output_dir results/allcat-extlarge-lr${lr} \
        --per_gpu_eval_batch_size=16 \
        --per_device_train_batch_size=16 \
        --overwrite_output
done

for lr in 5e-6 1e-5 2e-5; do
CUDA_VISIBLE_DEVICES=2 python -u main_allcat.py \
        --model_name_or_path  hfl/chinese-roberta-wwm-ext \
        --do_train --train_file CCPM-data/train.jsonl \
        --do_eval  --validation_file CCPM-data/split_valid.jsonl \
        --test_file CCPM-data/split_test.jsonl \
        --learning_rate ${lr}  --fp16 \
        --num_train_epochs 5 \
        --evaluation_strategy epoch \
        --output_dir results/allcat-ext-lr${lr} \
        --per_gpu_eval_batch_size=16 \
        --per_device_train_batch_size=16 \
        --overwrite_output
done

# CUDA_VISIBLE_DEVICES=2 python -u main_allcat.py \
#         --model_name_or_path  hfl/chinese-roberta-wwm-ext \
#         --do_train --train_file CCPM-data/train.jsonl \
#         --do_eval  --validation_file CCPM-data/split_valid.jsonl \
#         --test_file CCPM-data/split_test.jsonl \
#         --learning_rate 1e-5  --fp16 \
#         --num_train_epochs 3 \
#         --evaluation_strategy epoch \
#         --output_dir results/allcat \
#         --per_gpu_eval_batch_size=16 \
#         --per_device_train_batch_size=16 \
#         --overwrite_output