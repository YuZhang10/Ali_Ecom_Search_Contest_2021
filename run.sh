#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
DIR_PATH="./result/0403;simcse-roberta-pretain;bs128;nomlp;cls"

python train.py \
    --model_name_or_path cyclone/simcse-chinese-roberta-wwm-ext \
    --train_file "./data/X_train.csv" \
    --validation_file "./data/X_val.csv" \
    --output_dir $DIR_PATH \
    --num_train_epochs 2 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 3e-5 \
    --max_seq_length 64 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 50 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval
cp ./run.sh $DIR_PATH

echo "train finished!"

python 3.get_embedding.py --dir_path $DIR_PATH

bash 4.tar.sh
