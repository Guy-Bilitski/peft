#!/bin/bash

# Save this file as run_pissa_finetune.sh and make it executable with:
# chmod +x run_pissa_finetune.sh

export CUDA_VISIBLE_DEVICES=2,3   

deepspeed \
    GSM8_training.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --data_path metamath-dataset \
    --sub_task metamathqa:1000 \
    --dataset_field query response \
    --dataset_split train \
    --output_dir ./uilora-mistral-test \
    --full_finetune False \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_strategy no \
    --save_total_limit 1 \
    --eval_strategy no \
    --report_to none \
    --bf16 False \
    --fp16 False \
    --logging_dir ./logs \
    --do_train \
    --overwrite_output_dir \
    --deepspeed ds_config_zero3.json \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0