LOGFILE="output_$(date +'%Y%m%d_%H%M%S').log"

nohup python main_training.py \
    --epochs 60 \
    --seed 42 \
    --task sst2 \
    --num_svalues_to_adapt 128 \
    --num_svectors_to_adapt 100 \
    --head_lr 4e-3 \
    --adapter_lr 4e-3 \
    --batch_size 64 \
    --max_len 256 \
    --initial_scaler 0.1 \
    --initial_sigma 0.1 \
    --cuda_visible_devices 0 \
    --base_model_id roberta-base \
    --model_type uiortholora \
    --method_name UIOrthoLoRA \
    --uiortholora_alpha 16 \
    --uiortholora_dropout 0 \
    --target_modules q_proj v_proj \
    > "$LOGFILE" 2>&1 &

echo "Training launched. Logs: $LOGFILE"
