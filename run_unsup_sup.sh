#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
pretrain="cyclone/simcse-chinese-roberta-wwm-ext"
date=''
epoch=4
bs=128
pooler="cls"
max_seq_length=64
comment=''

model_name="${comment}_${date}_ep${epoch}_bs${bs}_${pooler}_max_seq_length${max_seq_length}"
dir_path="./result/${model_name}"
drive_result="/content/drive/MyDrive/competition/simcse-mini/result"

# 删除当前的文件夹
rm -rf $dir_path

# 无监督预训练
python my_train.py \
    --model_name_or_path $pretrain \
    --train_file  "./data/full_corpus_querys.csv"\
    --output_dir $dir_path/unsup \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --learning_rate 3e-5 \
    --max_seq_length $max_seq_length \
    --pooler_type $pooler \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
&& { echo "unsup train finished!"; } || { echo 'unsup train failed'; exit 1; }

# 清空显存
python clear_gpu_memory.py

# 有监督训练
python my_train.py \
    --model_name_or_path $dir_path/unsup \
    --train_file "./data/X_train.csv" \
    --validation_file "./data/X_val.csv" \
    --output_dir $dir_path/sup \
    --num_train_epochs $epoch \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --learning_rate 3e-5 \
    --max_seq_length $max_seq_length \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --greater_is_better False \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end \
    --eval_steps 100 \
    --save_steps 500 \
    --logging_steps 100 \
    --pooler_type $pooler \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
&& { echo "sup train finished!"; } || { echo 'sup train failed'; exit 1; }

# 保留原始脚本
cp ./run_unsup_sup.sh $dir_path/run_backup.sh

# 提取embedding
python get_embedding.py \
        --dir_path $dir_path/sup \
        --pooler_type $pooler \
        --temp 0.05 \
        --batchsize 1024 \
&& { echo "get embedding finished!"; } || { echo "get embedding failed"; exit 1; }

# 检查embedding文件
python data_check.py

# 打包embedding。放入result文件夹
tar zcvf foo.tar.gz query_embedding doc_embedding && mv foo.tar.gz $dir_path/${model_name}.tar.gz

# 将文件移回云盘保存
cp -r ./result/* $drive_result

echo "Finished!"
