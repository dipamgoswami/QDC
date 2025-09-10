# Query Drift Compensation for Continual Document Retrieval [![Paper](https://img.shields.io/badge/arXiv-2506.00037-brightgreen)](http://arxiv.org/abs/2506.00037)
Code for CoLLAs 2025 paper - Query Drift Compensation: Enabling Compatibility in Continual Learning of Retrieval Embedding Models

```
@inproceedings{goswami2025query,
  title={Query Drift Compensation: Enabling Compatibility in Continual Learning of Retrieval Embedding Models},
  author={Goswami, Dipam and Wang, Liying and Twardowski, Bartłomiej and van de Weijer, Joost},
  booktitle={Conference on Lifelong Learning Agents},
  year={2025},
}
```

# Abstract

Text embedding models enable semantic search, powering several NLP applications like Retrieval Augmented Generation by efficient information retrieval (IR). However, text embedding models are commonly studied in scenarios where the training data is static, thus limiting its applications to dy- namic scenarios where new training data emerges over time. IR methods generally encode a huge corpus of documents to low-dimensional embeddings and store them in a database index. During re- trieval, a semantic search over the corpus is performed and the document whose embedding is most similar to the query embedding is returned. When updating an embedding model with new training data, using the already indexed corpus is suboptimal due to the non-compatibility issue, since the model which was used to obtain the embeddings of the corpus has changed. While re-indexing of old corpus documents using the updated model enables compatibility, it requires much higher com- putation and time. Thus, it is critical to study how the already indexed corpus can still be effectively used without the need of re-indexing. In this work, we establish a continual learning benchmark with large-scale datasets and continually train dense retrieval embedding models on query-document pairs from new datasets in each task and observe forgetting on old tasks due to significant drift of embed- dings. We employ embedding distillation on both query and document embeddings to maintain stability and propose a novel query drift compensation method during retrieval to project new model query embeddings to the old embedding space. This enables compatibility with previously indexed corpus embeddings extracted using the old model and thus reduces the forgetting. We show that the proposed method significantly improves performance without any re-indexing.

<img src="https://github.com/dipamgoswami/QDC/blob/main/qdc.png" width="100%" height="100%">

# Model

We use the [nomic-bert-embed-v1-unsupervised](https://huggingface.co/nomic-ai/nomic-embed-text-v1-unsupervised) model provided by the [contrastors](https://github.com/nomic-ai/contrastors) repository from Nomic AI. The nomic embedding model is pre-trained with MLM followed by unsupervised contrastive pre-training. The nomic model is a modified version of BERT base resulting in a 137M parameter model with 8192 sequence length. Starting with the pre-trained model, we fine-tune them continually for our experiments.

# Environment
For setting up the environment, follow [contrastors](https://github.com/nomic-ai/contrastors).


# Data
For the first four datasets in the proposed CDR benchmark (MS Marco, NQ, HotpotQA, Fever), we take the datasets with already mined 7 hard-negatives per query from [contrastors](https://github.com/nomic-ai/contrastors). Follow the [Data Access](https://github.com/nomic-ai/contrastors?tab=readme-ov-file#data-access) instructions to download these datasets.

For FiQA2018, we use the [gte-base model](https://huggingface.co/thenlper/) to mine hard-negatives and we share the dataset [here](https://drive.google.com/drive/folders/15gEm7aBjFbdXZ6Achb67dBR9UPJWbf8p?usp=sharing).


# Run 

To run continual finetuning, follow the script [run_inc.sh](https://github.com/dipamgoswami/QDC/blob/main/run_inc.sh). 

# Eval 

Code for evaluation will be updated soon.

# Trained Checkpoints

We will release the trained models soon.
