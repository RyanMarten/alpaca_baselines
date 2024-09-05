
MODEL_PATH=/home/ec2-user/checkpoints/llama-7b
OUTPUT_DIR="dcft-alpaca-reproduction-ogsubset-10k-sft-EC2-8xA100-40GB"
MASTER_PORT=41592

mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=8 --master_port=$MASTER_PORT train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path ./alpaca_data_10k.json \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
