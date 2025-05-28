LOGFILE="output_$(date +'%Y%m%d_%H%M%S').log"

CUDA_VISIBLE_DEVICES=0 nohup python main_training.py \
    --epochs 60 \
    --seed 42 \
    --task sst2 \
    --num_svalues_to_adapt 128 \
    --num_svectors_to_adapt 60 \
    --head_lr 4e-4 \
    --adapter_lr 5e-3 \
    --batch_size 64 \
    --max_len 256 \
    --initial_scaler 0.01 \
    --initial_sigma 0.1 \
    --base_model_id roberta-base \
    --model_type uiortholora \
    --uiortholora_alpha 1 \
    --uiortholora_dropout 0 \
    --target_modules attention.output.dense query key value \
    --resume_from_checkpoint outputs/uiortholora_roberta-base_sst2/checkpoint-23166 \
    > "$LOGFILE" 2>&1 &

echo "Training launched. Logs: $LOGFILE"
