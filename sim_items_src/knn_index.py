import json
import os
import tarfile
from io import StringIO
import json
import time
from elasticsearch.exceptions import NotFoundError
from elasticsearch import Elasticsearch, helpers

import joblib
import boto3
import pandas as pd
import numpy as np
from numpy import dtype
import argparse

from config import (
    raw_feature_columns_names,
    raw_feature_columns_dtype,
    label_column,
    label_column_dtype,
    feature_columns_dtype,
    feature_columns,
    output_raw_columns,
    output_engineered_feature_columns,
)
from utils import get_es_client
from multiprocessing import Pool, TimeoutError
import uuid


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--opensearch_url", type=str, default=None)
    parser.add_argument("--opensearch_region", type=str, default=None)
    parser.add_argument("--opensearch_index_name", type=str, default=None)
    parser.add_argument("--opensearch_embeddings_dimension", type=int, default=64)
    parser.add_argument("--opensearch_ingestion_bz", type=int, default=32)
    parser.add_argument("--opensearch_num_threads", type=int, default=2)

    return parser.parse_known_args()


def get_data_from_a_folder(folder_name, suffix, dtype=None):
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [
        os.path.join(folder_name, file)
        for file in os.listdir(folder_name)
        if file.endswith(suffix)
    ]
    raw_data = [
        pd.read_csv(
            file,
            # header=None,
            names=output_raw_columns,
            # dtype={**feature_columns_dtype, **label_column_dtype}
        )
        for file in input_files
    ]
    data = pd.concat(raw_data).reset_index(drop=True)
    if dtype is not None:
        data = data.astype(dtype)
    return data


def get_embeddings_a_folder_of_jsonl(
    folder_name,
    suffix,
):
    # Take the set of files and read them all into a single pandas dataframe

    input_files = [
        os.path.join(folder_name, file)
        for file in os.listdir(folder_name)
        if file.endswith(suffix)
    ]
    data = []
    for file in input_files:
        with open(file) as f:
            for line in f:
                # each line is a list of json object
                for item in json.loads(line):
                    data.append(item["embedding"])
    return data


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"
    args, _ = parse_args()
    dataset = get_data_from_a_folder(
        f"{base_dir}/input_train",
        ".csv",
        # {**raw_feature_columns_dtype, **label_column_dtype},
    )
    embeddings = get_embeddings_a_folder_of_jsonl(f"{base_dir}/embeddings", ".out")

    print(f"Dataset shape: {dataset.shape}")
    print(f"Embeddings size: {len(embeddings)}")

    es = get_es_client(host=args.opensearch_url, region=args.opensearch_region)
    es.indices.delete(index=args.opensearch_index_name, ignore=[400, 404])

    try:
        es.indices.get_alias(args.opensearch_index_name)
        print(f"{args.opensearch_index_name} index exists!")
    except NotFoundError:
        index_settings = {
            "settings": {"index.knn": True, "index.knn.space_type": "cosinesimil"},
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": args.opensearch_embeddings_dimension,
                    }
                }
            },
        }

        es.indices.create(
            index=args.opensearch_index_name, body=json.dumps(index_settings)
        )
        print(f"Created {args.opensearch_index_name} index!")

    dataset = dataset.fillna("").assign(embeddings=embeddings)
    # can be optimized using bulk API
    num_threads = int(args.opensearch_num_threads)
    ingestion_bz = int(args.opensearch_ingestion_bz)

    if num_threads == 1 and ingestion_bz == 1:
        print("Ingest data with a single thread with index API")
        for idx, record in dataset.iterrows():
            body = record.to_dict()
            es.index(
                index=args.opensearch_index_name, id=idx, doc_type="_doc", body=body
            )
    else:
        print(f"Ingest data with {num_threads} threads with {ingestion_bz} bulk API")
        data_per_threads = np.array_split(
            dataset,
            num_threads,
        )

        def ingest(data_chunk):
            num_ingestion_chunk = len(data_chunk) // int(args.opensearch_ingestion_bz)
            print(
                f"num_ingestion_chunk: {num_ingestion_chunk}, opensearch_ingestion_bz: {args.opensearch_ingestion_bz}"
            )
            data_per_ingestion = np.array_split(
                data_chunk,
                num_ingestion_chunk,
            )
            print(f"data_per_ingestion sz: {len(data_per_ingestion[0])}")
            for chunk in data_per_ingestion:
                time.sleep(300 / 1000)
                actions = [
                    {
                        "_id": uuid.uuid4(),
                        "doc": record.to_dict(),
                    }  # random UUID for _id
                    for _, record in chunk.iterrows()
                ]
                helpers.bulk(
                    es,
                    actions=actions,
                    index=args.opensearch_index_name,
                    doc_type="_doc",
                )

        with Pool(processes=num_threads) as pool:
            pool.map(ingest, data_per_threads)

    print("Indexing completed!")
