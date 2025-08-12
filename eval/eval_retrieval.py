import logging
import math
import pickle
import cloudpickle
import multiprocess as mp  # uses dill instead of pickle -> required to load 2 models 

# import multiprocessing as mp

import os
import queue
from argparse import ArgumentParser

import numpy as np
import torch.nn as nn
import torch
import pickle
import dill
import re
from mteb import MTEB
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_distances

from contrastors.models.biencoder import BiEncoder, BiEncoderConfig

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")


TASK_LIST_RETRIEVAL = [
    "MSMARCO",
    "NQ",
    "HotpotQA",
    "FEVER",
    "FiQA2018",
]


class STransformer:
    def __init__(self, model, new_model, task, model_name, task_num, add_prefix=False): 
        self.model = model
        self.gpu_pool = self.model.start_multi_process_pool()
        self.new_model = new_model

        self.new_gpu_pool = self.new_model.start_multi_process_pool()
        
        self.add_prefix = add_prefix
        self.cbatch = 0
        model_name = re.split('/', model_name)[-2]
        self.corpus_dir = "corpus_cache/{}_corpus_{}".format(task, model_name)
        print("Storing corpus chunk embeddings in ",self.corpus_dir)

        if self.task_num > 1:
            with open('QDC_drift_vectors/query_drift_t1.pickle', 'rb') as handle:
                self.cluster_drift1 = pickle.load(handle)
                handle.close()
                self.cluster_drift = self.cluster_drift1
        if self.task_num > 2:
            with open('QDC_drift_vectors/query_drift_t2.pickle', 'rb') as handle:
                self.cluster_drift2 = pickle.load(handle)
                handle.close()
                self.cluster_drift = self.cluster_drift1 + self.cluster_drift2
        if self.task_num > 3:
            with open('QDC_drift_vectors/query_drift_t3.pickle', 'rb') as handle:
                self.cluster_drift3 = pickle.load(handle)
                handle.close()
                self.cluster_drift = self.cluster_drift1 + self.cluster_drift2 + self.cluster_drift3
        if self.task_num > 4:
            with open('QDC_drift_vectors/query_drift_t4.pickle', 'rb') as handle:
                self.cluster_drift4 = pickle.load(handle)
                handle.close()
                self.cluster_drift = self.cluster_drift1 + self.cluster_drift2 + self.cluster_drift3 + self.cluster_drift4


    def encode_queries(self, queries, **kwargs) -> np.ndarray:
        if self.add_prefix:
            input_texts = ['query: {}'.format(q) for q in queries]
        else:
            input_texts = queries

        pp = self.new_model.encode_multi_process(input_texts, self.new_gpu_pool, **kwargs)

        ## QDC with 1 drift vector ------------------------------------------
        pp = pp - np.array(self.cluster_drift.float())

        return pp

    def encode_corpus(self, corpus, **kwargs) -> np.ndarray:
        # with open('corpus.pickle', 'wb') as handle:
        #     pickle.dump("xyz", handle, protocol=pickle.HIGHEST_PROTOCOL)
        if isinstance(corpus[0], dict):
            input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        else:
            input_texts = corpus
        if self.add_prefix:
            input_texts = ['document: {}'.format(t) for t in input_texts]
        self.cbatch +=1
        
        # To store corpus embeddings in chunks
        pp = self.model.encode_multi_process(input_texts, self.gpu_pool, **kwargs)
        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir)
        with open('{}/chunk_{}.pickle'.format(self.corpus_dir, self.cbatch), 'wb') as handle:
            pickle.dump(pp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # To load stored corpus chunks
        # with open('{}/chunk_{}.pickle'.format(self.corpus_dir, self.cbatch), 'rb') as handle:
        #     pp = pickle.load(handle)
        #     handle.close()

        return pp


class CausalModel:
    def __init__(self, model_name): 
        config = BiEncoderConfig.from_pretrained(model_name)
        self.model = BiEncoder.from_pretrained(model_name, config=config)
        self.model.to(torch.bfloat16)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.model_max_length = 128

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, sentences, isquery, batch_size=256, **kwargs):
        embeddings = []

        device = kwargs.get("device", self.device)
        self.model  = self.model.to(device)

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                if encoded["input_ids"].shape[1] >= 2048 and batch_size > 256:
                    step_size = 128
                    for j in range(0, len(encoded["input_ids"]), step_size):
                        smaller_batch = {k: v[j : j + step_size].to(device) for k, v in encoded.items()}
                        curr_outputs = self.model(**smaller_batch)  #, normalize=True)

                        embeddings.extend(curr_outputs["embedding"].cpu().float().numpy())

                else:
                    outputs = self.model(**encoded.to(device))  #, normalize=True)
                    embeddings.extend(outputs["embedding"].cpu().float().numpy())

        return embeddings

    def start_multi_process_pool(self, isquery=False, target_devices=None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process

        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu'] * 4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(
                target=CausalModel._encode_multi_process_worker,
                args=(cuda_id, self, isquery, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, isquery, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                id, batch_size, sentences = input_queue.get()
                embeddings = model.encode(
                    sentences,
                    isquery,
                    device=target_device,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                )
                results_queue.put([id, embeddings])
            except queue.Empty:
                break

    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()

    def encode_multi_process(
        self,
        sentences,
        pool,
        isquery=False,
        batch_size=128,
        chunk_size=None,
        show_progress_bar=False,
        convert_to_numpy=None,
        convert_to_tensor=None,
    ):
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Numpy matrix with all embeddings
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.debug(f"Chunk data into {math.ceil(len(sentences) / chunk_size)} packages of size {chunk_size}")

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        embeddings = np.concatenate([np.array(t[1]) for t in results_list])
        return embeddings


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--old_model_name', type=str, required=True)    
    parser.add_argument('--new_model_name', type=str, required=True)
    parser.add_argument('--task_num', type=int, required=True)
    parser.add_argument("--add_prefix", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    old_model_name = args.old_model_name
    new_model_name = args.new_model_name
    task_num = args.task_num
    old_model = CausalModel(old_model_name)  
    new_model = CausalModel(new_model_name)
    

    for task in TASK_LIST_RETRIEVAL:
        logger.info(f"Running task: {task}")
        model2 = STransformer(old_model, new_model, task, old_model_name, task_num)

        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])  # Remove "en" for running all languages
        evaluation.run(model2, output_folder=f"results/{args.new_model_name}/task-id", eval_splits=eval_splits)
