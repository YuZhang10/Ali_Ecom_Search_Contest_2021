#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
DIR_PATH="./result/0402;simcse-chinese-roberta;bs128;nomlp;cls"
DRIVE_RESULT="/content/drive/MyDrive/competition/simcse-mini/result"

# 训练模型
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
echo "train finished!"

# 提取embedding
python 3.get_embedding.py --dir_path $DIR_PATH

# 打包embedding，放入result文件夹
bash 4.tar.sh
mv foo.tar.gz $DIR_PATH/foo.tar.gz

# 将文件移回云盘保存
cp -r ./result/* $DRIVE_RESULT

echo "Finished!"
