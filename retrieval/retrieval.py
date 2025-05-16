
import argparse

import os
import json
import time
from pprint import pprint

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from sentence_transformers import SentenceTransformer
from tqdm.std import tqdm

import pandas as pd
import numpy as np

from openai import OpenAI
from datasets import load_dataset



class ElasticSearch:
    def __init__(self, index, embedder='gpt'):
        """
        :param index: Elasticsearch index name.
        :param embedder: "gpt" uses OpenAI's text-embedding-3-small, "bge" uses SentenceTransformer's BAAI/bge-en-icl.
        """
        
        self.es = Elasticsearch('http://localhost:19200')
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)
        
        self.index = index
        self.embedder = embedder
        if self.embedder == "gpt":
            self.encoder = "text-embedding-3-small"
            self.client = OpenAI()  # Ensure your OpenAI API key is properly set in the environment.
        elif self.embedder == "bge":
            self.encoder = SentenceTransformer("BAAI/bge-en-icl")
        else:
            raise ValueError(f"Unknown embedder: {embedder}. Choose either 'gpt' or 'bge'.")
            
    def is_exist(self):
        if not self.es.indices.exists(index=self.index):
            raise ConnectionError(f"Index '{self.index}' does not exist.")
            
    def get_embedding(self, text):
        """
        Generate embedding for the given text using the selected embedding model.
        
        :param text: Input text string.
        :return: List of floats representing the embedding vector.
        """
        
        text = text.replace("\n", " ")
        if isinstance(self.encoder, str) and self.encoder.startswith("text-embedding"):
            response = self.client.embeddings.create(input=[text], model=self.encoder)
            output = response.data[0].embedding
        else:
            output = self.encoder.encode([text])[0].tolist()
        
        return output
    
    def search(self, **query_args):
        return self.es.search(index=self.index, **query_args)
    
    def retrieve_document(self, id):
        return self.es.get(index=self.index, id=id)

    def delete_index(self):
        self.es.indices.delete(index=self.index, ignore_unavailable=True)
        
    def create_index(self):
        self.es.indices.delete(index=self.index, ignore_unavailable=True)
        self.es.indices.create(index=self.index, body = {
            "settings": {
                "analysis": {
                    "char_filter": {
                        "my_char_filter": {
                            "type": "mapping",
                            "mappings": ["&=> and "]
                        }
                    },
                    "filter": {
                        "my_stop": {
                            "type": "stop",
                            "stopwords": "_english_"
                        },
                        "my_stemmer": {
                            "type": "stemmer",
                            "name": "english"
                        }
                    },
                    "analyzer": {
                        "my_analyzer": {
                            "type": "custom",
                            "char_filter": ["my_char_filter"],
                            "tokenizer": "standard",
                            "filter": ["lowercase", "my_stop", "my_stemmer"]
                        }
                    }
                }
            },
            'mappings':{
                'properties':{
                    'us_gaap_tag':{
                        'type': 'keyword'
                    },
                    'entity_type':{
                        'type': 'text'
                    },
                    'tag_text':{
                        'type': 'text',
                        "analyzer": "my_analyzer"
                    },
                    "bge_emb":{
                        'type': 'dense_vector',
                        'dims': 4096,
                        'similarity': 'cosine'
                    },
                    "gpt_emb":{
                        'type': 'dense_vector',
                        'dims': 1536,
                        'similarity': 'cosine'
                    }
                }
            }
        })
        
    def insert_document(self, document):
        return self.es.index(index=self.index, body=document)
    

    def insert_documents(self, documents, chunk_size=1000):
        def generate_operations(documents):
            for doc in documents:
                yield {'_index': self.index, '_source': doc}
        
        success, failed = 0, 0
        for ok, result in helpers.streaming_bulk(self.es, generate_operations(documents),\
                                                      chunk_size=chunk_size):
            if not ok:
                failed += 1
            else:
                success += 1
        return success, failed
    
    def reindex(self, json_file):
        self.create_index()
        with open(json_file, 'rt') as f:
            documents = json.loads(f.read())
        
        return self.insert_documents(documents)

class SearchModel:
    def __init__(self, search, search_mode="text"):
        self.search = search
        self.search_mode = search_mode
        
        if self.search.embedder == "gpt":
            self.emb_filed = "gpt_emb"
        else:
            self.emb_filed = "bge_emb"

    def searching(self,query, size=30):
        if self.search_mode == "text":
            return self.full_text_search(query, size)
        elif self.search_mode == "embedding":
            return self.embedding_search(query, size)
        elif self.search_mode == "hybrid":
            return self.hybrid_search(query, size)
        else:
            raise ValueError(f"Unknown search modes: {self.search_mode}. Choose 'text', 'embedding', or 'hybrid'.")

    def get_results(self, result):
        if result['hits']['total']['value'] > 0:
            res = []
            for hit in result['hits']['hits']:
                score = hit['_score']
                tag = hit['_source']['us_gaap_tag']
                type = hit['_source']['entity_type']
                text = hit['_source']['tag_text']
                res.append(f"us-gaap:{tag}")
            
            return res
        else:
            return []
            
    def embedding_search(self,query, size):
        result = self.search.search(
            knn = {
                'field': self.emb_filed,
                'query_vector': self.search.get_embedding(query),
                'num_candidates': 200,
                'k': size
            },
            size = size,
            from_ = 0
        )

        res = self.get_results(result)
        return res


    def full_text_search(self, query, size):
        result = self.search.search(
        
            query = {
                "bool":{
                    "should":[
                        {
                            "bool":{
                                "should":[
                                    {
                                        "match":{
                                            "entity_type":{
                                                "query": query,
                                                "boost": 1.0
                                            }
                                        }
                                    },
                                    {
                                        "match":{
                                            "tag_text":{
                                                "query": query,
                                                "boost": 1.0
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "fuzzy":{
                                "tag_text":{
                                    "value":query,
                                    "fuzziness": "AUTO",
                                    "boost":1.0
                                }
                            }
                        }
                    ]
                }
            },
            size = size,
            from_ = 0
        )
        res = self.get_results(result)
        return res


    def hybrid_search(self, query, size):
        result = self.search.search(
            query = {
                "bool":{
                    "should":[
                        {
                            "bool":{
                                "should":[
                                    {
                                        "match":{
                                            "entity_type":{
                                                "query": query,
                                                "boost": 1.0
                                            }
                                        }
                                    },
                                    {
                                        "match":{
                                            "tag_text":{
                                                "query": query,
                                                "boost": 1.0
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "fuzzy":{
                                "tag_text":{
                                    "value": query,
                                    "fuzziness": "AUTO",
                                    "boost":1.0
                                }
                            }
                        }
                    ]
                }
            },
            knn = {
                'field': self.emb_filed,
                'query_vector': self.search.get_embedding(query),
                'num_candidates': 200,
                'k': size
            },
            size = size,
            from_ = 0)

        res = self.get_results(result)
        return res

def upload_index_document(search, index_document):
    search.create_index()
    
    batch_size = 5000
    data_bacth = []
    with open(index_document,'r') as f:
        for i,item in tqdm(enumerate(f)):
            json_line = json.loads(item)
            data_bacth.append(json_line)
            if len(data_bacth) >= batch_size:
                search.insert_documents(data_bacth)
                data_bacth = []
        if data_bacth:
            search.insert_documents(data_bacth)
            data_bacth = []
    print('add all documents into database')

def retriever(search, input_data, size, search_mode="text"):
    """
        search_mode should be 'text', 'embedding', 'hybrid'
    """
    searcher = SearchModel(search, search_mode)
    result = []
    # input_data = input_data[:5]

    for i in tqdm(range(len(input_data))):
        query = input_data.at[i, "query"]
        res = searcher.searching(query, size)
        result.append({"query": query, "prediction": res})
    
    return result

def dcg_at_k(rel, k):
    """Compute DCG@k given binary relevance list"""
    rel = rel[:k]
    return sum(r / np.log2(i + 2) for i, r in enumerate(rel))

def ndcg_at_k(pred, gold, k):
    """Normalized DCG@k"""
    rel = [1 if p == gold else 0 for p in pred]
    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0

def hit_at_k(pred, gold, k):
    """Hit@K (1 if gold appears in top-K, else 0)"""
    return int(gold in pred[:k])

def reciprocal_rank(pred, gold):
    """Reciprocal rank"""
    try:
        rank = pred.index(gold) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0

def evaluate_metrics(predictions, references, ks=[1, 2, 3]):
    """
    Args:
        predictions: List[List[str]]  - ranked candidate doc ids for each query
        references: List[List[str]]   - each query's correct doc(s); assumes single correct doc per query
        ks: List[int]                 - cutoffs for Hit@K and nDCG@K

    Returns:
        dict with MRR, Hit@K, nDCG@K
    """
    eval_result = {}
    num_queries = len(predictions)

    mrr_total = 0.0
    hit_at_k_total = {k: 0 for k in ks}
    ndcg_at_k_total = {k: 0.0 for k in ks}

    for pred, ref_list in tqdm(zip(predictions, references)):
        gold = ref_list[0]  # assume single correct doc

        # MRR
        mrr_total += reciprocal_rank(pred, gold)

        for k in ks:
            hit_at_k_total[k] += hit_at_k(pred, gold, k)
            ndcg_at_k_total[k] += ndcg_at_k(pred, gold, k)

    eval_result["MRR"] = round(mrr_total / num_queries, 4)

    for k in ks:
        eval_result[f"Hit@{k}"] = round(hit_at_k_total[k] / num_queries, 4)
        eval_result[f"nDCG@{k}"] = round(ndcg_at_k_total[k] / num_queries, 4)

    return eval_result

def parse_input(dataset_name: str) -> pd.DataFrame:
    if os.path.exists(dataset_name):
        # local path
        print(f"Loading local dataset from {dataset_name}...")
        input_data = None
        if dataset_name.endswith("csv"):
            input_data = pd.read_csv(dataset_name)
        elif dataset_name.endswith("tsv"):
            input_data = pd.read_csv(dataset_name, sep="\t")
        elif dataset_name.endswith("json"):
            with open(dataset_name, "r") as f:
                input_data = json.load(f)
            input_data = pd.DataFrame(input_data)
        else:
            raise ValueError(f"Unknown data format: {dataset_name}. Choose 'CSV', 'TSV', or 'JSON' format.")
        return input_data
    else:
        # Hugging Face Hub 
        print(f"Loading dataset from Hugging Face Hub: {dataset_name}")
        input_data = load_dataset(dataset_name)
        return input_data["test"].to_pandas()
    

def write_file(res, output_path, output_name):
    final_path = os.path.join(output_path,output_name)
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    
    with open(final_path, "w") as f:
        json.dump(res, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Run XBRL Tagging Entity Normalization Retrieval")
    parser.add_argument("--dataset_name", type=str, default="TheFinAI/XBRL_Tagging_Entity_Normalization",
                        help="Dataset name to load input from")
    parser.add_argument("--output_file", type=str, default="./output",
                        help="Directory to save the output results")
    parser.add_argument("--size", type=int, default=100,
                        help="Number of top documents to retrieve")
    parser.add_argument("--search_mode", type=str, choices=["text", "embedding", "hybrid"], default="embedding",
                        help="Search mode to use for retrieval")
    parser.add_argument("--embedder_name", type=str, choices=["gpt", "bge"], default="gpt",
                        help="Embedder type to use")
    parser.add_argument("--index_name", type=str, default="us_gaap_taxonomy",
                        help="Name of the Elasticsearch index")
    parser.add_argument("--index_document", type=str, default="./index_document/us_gaap_2024_final_embedder.jsonl",
                        help="Path to index document for Elasticsearch")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode")
    parser.add_argument("--ks", type=str, default="1,5,10,20",
                        help="Comma-separated list of cutoff values for evaluation (e.g., 1,5,10)")
    

    args = parser.parse_args()
    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip().isdigit()]

    # load the input data
    input_data = parse_input(args.dataset_name)

    # load elasticsearch model
    search = ElasticSearch(args.index_name, embedder=args.embedder_name)

    try:
        search.is_exist()
    except Exception as e:
        print(f"Error initializing search module: {e}")
        print("start to upload the index document......")
        if not os.path.exists(args.index_document):
            print("You don't have this index document in your ElasticSearch, please upload your index document first.")
            return
        upload_index_document(search, args.index_document)

    # start to retrieve
    result = retriever(search, input_data, args.size, search_mode=args.search_mode)
    
    final_result = None
    # evaluating
    if args.eval:
        final_result = []
        print("start to evaluate the results......")
        references = [[tag] for tag in input_data.answer.tolist()]
        prediction = [pre.get("prediction") for pre in result]
        evaluate_res = evaluate_metrics(prediction, references, ks=ks)
        print(evaluate_res)

        ## add the ground truth answer to prediction results
        for pre, tag in zip(result, input_data.answer.tolist()):
            pre["answer"] = tag
            final_result.append(pre)
    
    # saving to output file
    print("start to save the results")

    dataset_n = args.dataset_name.split("/")[-1]
    if len(dataset_n.split("."))>1:
        dataset_n = dataset_n.split(".")[0]
        
    output_name = None
    evaluated_name = None
    if args.search_mode == "text":
        output_name=f"retrieval_result_{dataset_n}_{args.search_mode}.json"
        evaluated_name=f"evaluated_result_{dataset_n}_{args.search_mode}.json"
    else:
        output_name=f"retreval_result_{dataset_n}_{args.search_mode}_{args.embedder_name}.json"
        evaluated_name=f"evaluated_result_{dataset_n}_{args.search_mode}_{args.embedder_name}.json"

    if args.eval:
        write_file(final_result, args.output_file, output_name)
        write_file(evaluate_res, args.output_file, evaluated_name)
    else:
        write_file(result, args.output_file, output_name)
    print("saving results successfully")

if __name__ == "__main__":
    main()
