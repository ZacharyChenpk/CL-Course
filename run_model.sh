CUDA_VISIBLE_DEVICES=3 python -u main.py \
        --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
        --do_train --train_file CCPM-data/train.jsonl \
        --do_eval  --validation_file CCPM-data/split_valid.jsonl \
        --test_file CCPM-data/split_test.jsonl \
        --learning_rate 1e-5  --fp16 \
        --num_train_epochs 10 \
        --softmax_temperature 1 \
        --multiplier 1 \
        --tagging_ratio 0.5 \
        --evaluation_strategy epoch \
        --output_dir results/allcattag_extlarge_lr1e-5_tmpr1_mtply1_tr0.5_nte10_bsz16 \
        --per_gpu_eval_batch_size=16 \
        --per_device_train_batch_size=16 \
        --save_steps 2724 \
        --overwrite_output