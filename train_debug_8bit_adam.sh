./venv/bin/torchrun --nproc_per_node=auto --master_port=12345 train.py \
    --model_name_or_path $WORK/dcft/llama-7b \
    --data_path ./alpaca_data.json \
    --bf16 False \
    --gradient_checkpointing True \
    --output_dir $WORK/dcft/llama-7b-checkpoints \
    --include_num_input_tokens_seen \
    --include_tokens_per_second \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --optim "adamw_bnb_8bit" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
    # --fsdp "full_shard auto_wrap offload" \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 1 \