import os
import torch
import logging
from sagemaker_huggingface_inference_toolkit import content_types, decoder_encoder

import torch.nn.functional as F

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if_knn_search = os.getenv("HF_KNN", "false") == "true"
    if if_knn_search:
        from utils import get_es_client

        es_client = get_es_client(
            host=os.getenv("ES_HOST", None),
            region=os.getenv("ES_REGION", None),
        )
        k = int(os.getenv("ES_K", "20"))
        for instance in prediction:
            body = {
                "size": k,
                "_source": {
                    "exclude": ["embeddings"],
                },
                "query": {
                    "knn": {
                        "embeddings": {
                            "vector": instance["embedding"],
                            "k": k,
                        }
                    }
                },
            }
            res = es_client.search(index=os.getenv("ES_INDEX_NAME", None), body=body)
            similar_items = res["hits"]["hits"]
            instance["similar_items"] = similar_items
    else:
        pass
    return decoder_encoder.encode(prediction, accept)


def predict_fn(data, hf_pipeline):
    # destruct model and tokenizer

    # pop inputs for pipeline
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", {})

    if_extract_emb = os.getenv("HF_EMB", "false") == "true"

    activation = {"embeddings": []}
    if if_extract_emb:

        def get_activation(name, embeddings):
            def hook(model, input, output):
                activation[name].append(output)

            return hook

        hf_pipeline.model.pre_classifier.register_forward_hook(
            get_activation("embeddings", activation)
        )

    prediction = hf_pipeline(inputs, **parameters)

    if if_extract_emb:
        sentence_embeddings = torch.cat(activation["embeddings"])
        # logger.info(f"sentence_embeddings shape: {sentence_embeddings.shape}")
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # logger.info(f"af sentence_embeddings shape: {sentence_embeddings.shape}")
        sentence_embeddings = sentence_embeddings.numpy().tolist()

        # logger.info(f"sentence_embeddings: {len(sentence_embeddings)}")
        return [
            {**pred, "embedding": embed}
            for pred, embed in zip(prediction, sentence_embeddings)
        ]
    else:
        return prediction
