import logging
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor, device
import transformers
from .models import BertForCL, RobertaForCL
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple, Type, Union
from collections import namedtuple
import pandas as pd
import argparse
import collections

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SimCSE(object):
    """
    A class for embedding sentences, calculating similarities, and retriving sentences by SimCSE.
    """

    def __init__(self, model_name_or_path: str,
                 device: str = None,
                 num_cells: int = 100,
                 num_cells_in_search: int = 10,
                 pooler="cls"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model_args = namedtuple(
            "model_args", ["do_mlm", "pooler_type", "temp", "mlp_only_train", "do_ema"])
        dummy_args = model_args(False,
                                pooler,
                                '0.05',
                                False,
                                False)
        print(dummy_args)
        self.model = BertForCL.from_pretrained(model_name_or_path,
                                               model_args=dummy_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        self.pooler = pooler

    def encode(self, sentence: Union[str, List[str]],
               device: str = None,
               return_numpy: bool = False,
               normalize_to_unit: bool = True,
               keepdim: bool = False,
               batch_size: int = 64,
               max_length: int = 128) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + \
                (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True, sent_emb=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                if normalize_to_unit:
                    embeddings = embeddings / \
                        embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings

    def similarity(self, queries: Union[str, List[str]],
                   keys: Union[str, List[str], ndarray],
                   device: str = None) -> Union[float, ndarray]:

        query_vecs = self.encode(
            queries, device=device, return_numpy=True)  # suppose N queries

        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, device=device,
                                   return_numpy=True)  # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = len(
            query_vecs.shape) == 1, len(key_vecs.shape) == 1
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)

        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)

        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])

        return similarities

    def build_index(self, sentences_or_file_path: Union[str, List[str]],
                    use_faiss: bool = None,
                    faiss_fast: bool = False,
                    device: str = None,
                    batch_size: int = 64):

        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True
            except:
                logger.warning(
                    "Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False

        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." %
                             (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences

        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device,
                                 batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

        logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}

        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(
                    self.num_cells, len(sentences_or_file_path)))
            else:
                index = quantizer

            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(20 * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    logger.info("Use CPU-version faiss")
            else:
                logger.info("Use CPU-version faiss")

            if faiss_fast:
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search,
                               len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")

    def search(self, queries: Union[str, List[str]],
               device: str = None,
               threshold: float = 0.6,
               top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:

        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device)
                    combined_results.append(results)
                return combined_results

            similarities = self.similarity(
                queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(
                id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["sentences"][idx], score)
                       for idx, score in id_and_score]
            return results
        else:
            query_vecs = self.encode(
                queries, device=device, normalize_to_unit=True, keepdim=True, return_numpy=True)

            distance, idx = self.index["index"].search(
                query_vecs.astype(np.float32), top_k)

            def pack_single_result(dist, idx):
                results = [(self.index["sentences"][i], s)
                           for i, s in zip(idx, dist)]
                return results

            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])

    def calc_MRR(self, querys: List[str], gnames: List[str], bad_case_file_name=None):
        '''
        wentian match metric
        '''
        bad_qs = []
        bad_gs = []
        bad_results = []
        MRRs = []
        self.build_index(
            gnames, use_faiss=True, batch_size=512, device=self.device)
        results = self.search(querys, top_k=10)
        for q, g, result in tqdm(zip(querys, gnames, results)):
            idx = -1
            for i, (r, s) in enumerate(result):
                if r == g:
                    idx = i+1
                    break
            if idx == -1:
                bad_qs.append(q)
                bad_gs.append(g)
                bad_results.append(result)
                MRRs.append(0)
            else:
                MRRs.append(1/idx)
        failed_cnt = sum([1 for m in MRRs if m == 0])
        MRR = sum(MRRs)/len(querys)
        logger.info(f"Failed to find {failed_cnt} query")
        logger.info(f"MRR = {MRR}")
        if bad_case_file_name:
            pd.DataFrame({'query': bad_qs, 'gnames': bad_gs, 'bad_result': bad_results}).to_csv(
                bad_case_file_name, index=False)
        return MRRs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', type=str, default="./result/sup-simcse/")
    args = parser.parse_args()
    print(args)

    model = SimCSE(args.model_path)
    val = pd.read_csv("../data/simple_X_val.csv")
    querys, gnames = val['query'].values.tolist(), val['gname'].values.tolist()
    MRRs = model.calc_MRR(querys, gnames)
    logger.info(f"MRRs counter = {collections.Counter(MRRs)}")

