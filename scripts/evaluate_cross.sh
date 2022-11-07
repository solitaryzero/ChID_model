CUDA_VISIBLE_DEVICES=1 python ./src/evaluate.py \
    --data_path ./data/ChID \
    --model_path ./models/cross/10w \
    --model_mode cross \
    --data_split test \
    --seed 42 \
    --cuda