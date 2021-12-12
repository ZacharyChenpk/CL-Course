for lr in 2e-5 5e-5 1e-4; do
    for nte in 2 3 4; do
        for bsz in 8 16 24; do
CUDA_VISIBLE_DEVICES=2 python -u main.py \
          --model_name_or_path  hfl/chinese-roberta-wwm-ext-large \
          --do_train --train_file CCPM-data/train.jsonl \
          --do_eval  --validation_file CCPM-data/split_valid.jsonl \
          --test_file CCPM-data/split_test.jsonl \
          --learning_rate ${lr}  --fp16 \
          --num_train_epochs ${nte} \
          --output_dir results/ext-large_lr${lr}_nte${nte}_bsz${bsz} \
          --per_gpu_eval_batch_size=16 \
          --per_device_train_batch_size=${bsz} \
          --overwrite_output \
        #   > ext-large.log 2>&1 &
        done
    done
done