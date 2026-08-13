from rag import rag
from search import (
    load_docs,
    load_embedding_model,
    load_embeddings,
    DOCUMENTS_FILE,
    EMBEDDINGS_FILE,
)

import time

queries = [
    "Who directed Oppenheimer?",
    "Who directed Inception?",
    "Who directed Barbie?",
    "Who directed Dune?",
    "Who directed Dune: Part Two?",
    "Who directed Get Out?",
    "Who stars in A Man Called Otto?",
    "Who stars in Greyhound?",
    "Who stars in The Revenant?",
    "Who stars in Killers of the Flower Moon?",
    "Tell me about Oppenheimer",
    "Tell me about Barbie",
    "Tell me about Greyhound",
    "Tell me about Once Upon a Time in Hollywood",
    "Tell me about Killers of the Flower Moon",
    "What is Parasite about?",
    "What is Blade Runner 2049 about?",
    "Tell me about Everything Everywhere All at Once",
    "List 5 movies with Tom Hanks",
    "List 5 movies with Leonardo DiCaprio",
]

docs = load_docs(DOCUMENTS_FILE)
model = load_embedding_model()
embeddings = load_embeddings(EMBEDDINGS_FILE)

for query in queries:
    print(f"\nQUERY: {query}")

    answer = rag(
        query=query,
        model=model,
        embeddings=embeddings,
        documents=docs,
    )

    print(f"ANSWER: {answer}")

print("\nTest finished. Keeping metrics server alive...")
while True:
    time.sleep(60)