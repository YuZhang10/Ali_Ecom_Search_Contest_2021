#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
pretrain="cyclone/simcse-chinese-roberta-wwm-ext"
date='0403'
epoch=2
bs=64
pooler="cls"
max_seq_length=64

dir_path="./result/${date};ep${epoch};bs${bs};${pooler};max_seq_length${max_seq_length};"
drive_result="/content/drive/MyDrive/competition/simcse-mini/result"

# 训练模型
python 2.train.py \
    --model_name_or_path $pretrain \
    --train_file "./data/X_train.csv" \
    --validation_file "./data/X_val.csv" \
    --output_dir $dir_path \
    --num_train_epochs $epoch \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 3e-5 \
    --max_seq_length $max_seq_length \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --greater_is_better False \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end \
    --eval_steps 5 \
    --save_steps 5 \
    --logging_steps 5 \
    --pooler_type $pooler \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval
echo "train finished!"

# 保留原始脚本
cp ./run.sh $dir_path/run_backup.sh

# 提取embedding
python 3.get_embedding.py --dir_path $dir_path

# 打包embedding，放入result文件夹
tar zcvf $dir_path/foo.tar.gz  \
$dir_path/query_embedding \
$dir_path/doc_embedding

# 将文件移回云盘保存
cp -r ./result/* $drive_result

echo "Finished!"
