# CUDA_VISIBLE_DEVICES=1 python ./src/train.py \
#     --data_path ./data/ChID \
#     --output_path ./models \
#     --base_model hfl/chinese-roberta-wwm-ext \
#     --train_data_num 10w \
#     --model_mode cross \
#     --max_seq_length 512 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --epoch 4 \
#     --warmup_proportion 0.1 \
#     --train_batch_size 32 \
#     --eval_batch_size 32 \
#     --print_interval 5 \
#     --eval_interval 2000 \
#     --shuffle \
#     --do_evaluate \
#     --final_evaluate \
#     --seed 42 \
#     --cuda

# CUDA_VISIBLE_DEVICES=1 python ./src/train.py \
#     --data_path ./data/ChID \
#     --output_path ./models \
#     --base_model hfl/chinese-electra-180g-large-discriminator \
#     --train_data_num 10w \
#     --model_mode cross \
#     --max_seq_length 512 \
#     --learning_rate 1e-5 \
#     --max_grad_norm 1.0 \
#     --epoch 4 \
#     --warmup_proportion 0.1 \
#     --train_batch_size 8 \
#     --eval_batch_size 32 \
#     --print_interval 5 \
#     --eval_interval 2000 \
#     --shuffle \
#     --do_evaluate \
#     --final_evaluate \
#     --seed 42 \
#     --cuda

CUDA_VISIBLE_DEVICES=1 python ./src/train.py \
    --data_path ./data/ChID \
    --output_path ./models \
    --base_model hfl/chinese-electra-180g-large-discriminator \
    --train_data_num 10w \
    --model_mode cross \
    --max_seq_length 256 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --epoch 4 \
    --warmup_proportion 0.1 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --print_interval 5 \
    --eval_interval 2000 \
    --shuffle \
    --do_evaluate \
    --final_evaluate \
    --seed 42 \
    --cuda