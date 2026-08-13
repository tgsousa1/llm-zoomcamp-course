import json
import numpy as np

from pathlib import Path
from minsearch import Index
from sentence_transformers import SentenceTransformer

from monitoring import tracer

PROCESSED_DATA_PATH = Path("data/processed")
DOCUMENTS_FILE = PROCESSED_DATA_PATH / "documents.json"
EMBEDDINGS_FILE = PROCESSED_DATA_PATH / "embeddings.npy"

def load_docs(docs_file):
    with open(docs_file, "r",encoding="utf-8") as f:
        documents=json.load(f)
    
    return documents

def build_text_index(documents):
    index=Index(
        text_fields=["content"],
        keyword_fields=["id"]
    )
    index.fit(documents)

    return index

def load_embedding_model(model_name="multi-qa-MiniLM-L6-cos-v1"):
    return SentenceTransformer(model_name)

def build_embeddings(documents,model):
    txts= [doc["content"] for doc in documents]

    embeddings=model.encode(
        txts,
        show_progress_bar=True
    )

    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    np.save(output_path, embeddings)

def load_embeddings(input_path):
    return np.load(input_path)

def vector_search(query, model, embeddings, documents, num_results=5):

    with tracer.start_as_current_span("vector_search") as span:

        span.set_attribute("query", query)
        span.set_attribute("num_documents", len(documents))
        span.set_attribute("num_results", num_results)

        query_vector = model.encode(query)

        scores = embeddings.dot(query_vector)

        top_idx = np.argsort(scores)[::-1][:num_results]

        results = [documents[i] for i in top_idx]

        span.set_attribute("results_count",len(results))

        return results

def text_search(index, query, num_results=5):
    boost_dict={"content":1.5}

    return index.search(
        query=query,
        boost_dict=boost_dict,
        num_results=num_results
    )

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc["id"]

            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)

    return [docs[key] for key in ranked[:num_results]]

def hybrid_search(query, index, model, embeddings, documents, num_results=5):
    text_results = text_search(index, query, num_results)

    vector_results = vector_search(
        query,
        model,
        embeddings,
        documents,
        num_results
    )

    return rrf([text_results, vector_results], num_results=num_results)

###testing
if __name__ == "__main__":

    docs = load_docs(DOCUMENTS_FILE)

    text_index = build_text_index(docs)

    model = load_embedding_model()

    if EMBEDDINGS_FILE.exists():
        embeddings = load_embeddings(EMBEDDINGS_FILE)
    else:
        embeddings = build_embeddings(docs, model)
        save_embeddings(embeddings, EMBEDDINGS_FILE)

    query = "movies directed by Christopher Nolan"

    results = hybrid_search(
        query,
        text_index,
        model,
        embeddings,
        docs,
        num_results=5
    )

    for movie in results:
        print(movie["title"])
