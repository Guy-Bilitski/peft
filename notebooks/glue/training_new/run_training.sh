LOGFILE="output_$(date +'%Y%m%d_%H%M%S').log"

CUDA_VISIBLE_DEVICES=1 nohup python main_training.py \
    --epochs 2 \
    --seed 42 \
    --task cola \
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
    > "$LOGFILE" 2>&1 &

echo "Training launched. Logs: $LOGFILE"
