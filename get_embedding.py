import csv
import sys
import argparse
import os
from tkinter.font import BOLD
from numpy import bool_
import pandas as pd

import torch
from tqdm import tqdm

sys.path.append("..")
from my_simcse.models import BertForCL, RobertaForCL
from transformers import AutoTokenizer
from collections import namedtuple
device = "cuda:0"
use_pinyin = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir_path', type=str, default="./result/sup-simcse/")
    parser.add_argument('--do_mlm', action='store_true', default=False)
    parser.add_argument('--pooler_type', type=str, default="cls")
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--mlp_only_train', action='store_true', default=False)
    parser.add_argument('--do_ema', action='store_true', default=False)
    parser.add_argument('--batchsize', type=int, default=1024)
    parser.add_argument('--max_seq_length', type=int, default=64)
    args = parser.parse_args()
    print(args)
    model_args = namedtuple("model_args",["do_mlm","pooler_type","temp","mlp_only_train","do_ema"])
    dummy_args = model_args(args.do_mlm,
                            args.pooler_type, 
                            args.temp,
                            args.mlp_only_train,
                            args.do_ema)
    print(dummy_args)
    model = BertForCL.from_pretrained(args.dir_path,
                                      model_args=dummy_args)
    model.to(device)
    corpus = pd.read_csv("./data/simple_corpus.csv")['gname'].values.tolist()
    query = pd.read_csv("./data/simple_test_query.csv")['query'].values.tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.dir_path)

    def encode_fun(texts, model):
        inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=args.max_seq_length)
        inputs.to(device)
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
            embeddings = embeddings.squeeze(0).cpu().numpy()
        return embeddings

    print("Processing query embedding...")
    query_embedding_file = csv.writer(open('query_embedding', 'w'), delimiter='\t')
    batch_size = args.batchsize
    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        assert temp_embedding.shape[-1]==128, f"embedding dim {temp_embedding.shape[-1]} != 128!"
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            query_embedding_file.writerow([i + j + 200001, writer_str])

    print("Processing doc embedding...")
    doc_embedding_file = csv.writer(open('doc_embedding', 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
        temp_embedding = encode_fun(batch_text, model)
        assert temp_embedding.shape[-1]==128, f"embedding dim {temp_embedding.shape[-1]} != 128!"
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            doc_embedding_file.writerow([i + j + 1, writer_str])

    