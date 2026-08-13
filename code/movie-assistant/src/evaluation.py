from search import(
    load_docs, build_text_index, load_embedding_model, load_embeddings, 
    text_search, vector_search, hybrid_search, DOCUMENTS_FILE, EMBEDDINGS_FILE
)
from rag import build_context
from openai import OpenAI

client = OpenAI()


####SEARCH EVALUATION##

test_queries = [
    {
        "query": "Who directed Oppenheimer?",
        "expected_title": "Oppenheimer"
    },
    {
        "query": "Who directed Inception?",
        "expected_title": "Inception"
    },
    {
        "query": "Who directed Barbie?",
        "expected_title": "Barbie"
    },
    {
        "query": "Tell me about Killers of the Flower Moon",
        "expected_title": "Killers of the Flower Moon"
    },
    {
        "query": "Tell me about Once Upon a Time in Hollywood",
        "expected_title": "Once Upon a Time in Hollywood"
    },
    {
        "query": "Tell me about Greyhound",
        "expected_title": "Greyhound"
    },
    {
        "query": "Who directed Dune?",
        "expected_title": "Dune"
    },
    {
        "query": "Who directed Dune: Part Two?",
        "expected_title": "Dune: Part Two"
    },
    {
        "query": "Tell me about Everything Everywhere All at Once",
        "expected_title": "Everything Everywhere All at Once"
    },
    {
        "query": "What is Parasite about?",
        "expected_title": "Parasite"
    },
    {
        "query": "Who directed Get Out?",
        "expected_title": "Get Out"
    },
    {
        "query": "Tell me about Spider-Man: Into the Spider-Verse",
        "expected_title": "Spider-Man: Into the Spider-Verse"
    },
    {
        "query": "Who directed Once Upon a Time in Hollywood?",
        "expected_title": "Once Upon a Time in Hollywood"
    },
    {
        "query": "What is Blade Runner 2049 about?",
        "expected_title": "Blade Runner 2049"
    },
    {
        "query": "Tell me about Ford v Ferrari",
        "expected_title": "Ford v Ferrari"
    }   
]

llm_test_queries = [
    # Factual questions
    {
        "query": "Who directed The Astronaut?",
        "expected_answer": "Jess Varley"
    },
    {
        "query": "Who wrote Believe?",
        "expected_answer": "Billy Dickson"
    },
    {
        "query": "When was Ben-Hur released?",
        "expected_answer": "August 9, 2016"
    },
    {
        "query": "What language is Pilot in?",
        "expected_answer": "Korean"
    },
    {
        "query": "Who stars in Tau?",
        "expected_answer": "Maika Monroe"
    },

    # Contextual / multi-field questions
    {
        "query": "Who directed and wrote The Astronaut?",
        "expected_answer": "Jess Varley"
    },
    {
        "query": "What is The Astronaut about?",
        "expected_answer": "extraterrestrial entity"
    },
    {
        "query": "Who plays the main character in Ben-Hur?",
        "expected_answer": "Jack Huston"
    },
    {
        "query": "What happens to the main character in Pilot?",
        "expected_answer": "becomes unemployed"
    },
    {
        "query": "Who directed and wrote The Moogai?",
        "expected_answer": "Jon Bell"
    },

    # Information not present in the context
    {
        "query": "How many Oscars did The Astronaut win?",
        "expected_answer": "I don't know"
    },
    {
        "query": "What was the budget of Believe?",
        "expected_answer": "I don't know"
    },
    {
        "query": "How many Academy Awards did Ben-Hur win?",
        "expected_answer": "I don't know"
    },
    {
        "query": "What was the box office revenue of Tau?",
        "expected_answer": "I don't know"
    },
    {
        "query": "How many awards did Pilot win?",
        "expected_answer": "I don't know"
    }
]

def hit_rate(results, expected_title):
    for result in results:
        if result["title"] == expected_title:
            return 1

    return 0

def evaluate_text_search(test_queries, index):
    hits = 0
    print("\nHit Rate - Text Search\n")
    for test in test_queries:
        results = text_search(
            index,
            test["query"],
            num_results=5
        )

        hit = hit_rate(results, test["expected_title"])
        hits += hit

        print(
            test["query"],
            "->",
            test["expected_title"],
            "HIT" if hit else "MISS"
        )

    return hits / len(test_queries)

def evaluate_vector_search(test_queries, model, embeddings, docs):
    hits = 0

    print("\nHit Rate - Vector Search\n")
    for test in test_queries:
        results = vector_search(
            test["query"],
            model,
            embeddings,
            docs,
            num_results=5
        )

        hit = hit_rate(results, test["expected_title"])
        hits += hit

        print(
            test["query"],
            "->",
            test["expected_title"],
            "HIT" if hit else "MISS"
        )

    return hits / len(test_queries)

def evaluate_hybrid_search(test_queries, index, model, embeddings, docs):
    hits = 0
    
    print("\nHit Rate - Hybrid Search\n")
    for test in test_queries:
        results = hybrid_search(
            test["query"],
            index,
            model,
            embeddings,
            docs,
            num_results=5
        )

        hit = hit_rate(results, test["expected_title"])
        hits += hit

        print(
            test["query"],
            "->",
            test["expected_title"],
            "HIT" if hit else "MISS"
        )

    return hits / len(test_queries)

def reciprocal_rank(results, expected_title):
    for rank, result in enumerate(results, start=1):
        if result["title"] == expected_title:
            return 1 / rank

    return 0

def calculate_mrr(scores):
    return sum(scores) / len(scores)

def evaluate_text_mrr(test_queries, index):
    scores=[]

    for test in test_queries:
        results=text_search(
            index,
            test["query"],
            num_results=5
        )

        score=reciprocal_rank(
            results,
            test["expected_title"]
        )

        scores.append(score)

    return calculate_mrr(scores)

def evaluate_vector_mrr(test_queries, model, embeddings, docs):
    scores = []

    for test in test_queries:
        results=vector_search(
            test["query"],
            model,
            embeddings,
            docs,
            num_results=5
        )

        score = reciprocal_rank(
            results,
            test["expected_title"]
        )

        scores.append(score)

    return calculate_mrr(scores)

def evaluate_hybrid_mrr(test_queries, index, model, embeddings, docs):
    scores = []

    for test in test_queries:
        results=hybrid_search(
            test["query"],
            index,
            model,
            embeddings,
            docs,
            num_results=5
        )

        score=reciprocal_rank(
            results,
            test["expected_title"]
        )

        scores.append(score)

    return calculate_mrr(scores)


###LLM EVALUATION###
PROMPT_A = """
Answer the QUESTION using the CONTEXT below.

QUESTION:
{question}

CONTEXT:
{context}
"""

PROMPT_B = """
You are a movie information assistant.

Answer the user's question using ONLY the provided context.

- Pay attention to the keywords in the user's query.
- For questions about actors, actresses, or directors, look for the relevant information in the movie content.
- If the answer cannot be found in the context, say that you don't know.
- Don't suggest movies with similar titles.
- Keep your answers concise and factual.

QUESTION:
{question}

CONTEXT:
{context}
"""

def generate_answer(query, results, prompt_template):
    context = build_context(results)

    prompt = prompt_template.format(
        question=query,
        context=context
    )

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    return response.output_text

def answer_is_correct(answer, expected_answer):
    answer = answer.lower()

    if expected_answer == "I don't know":
        unknown_phrases = [
            "i don't know",
            "not provided",
            "not mentioned",
            "not stated",
            "does not provide",
            "cannot be determined",
            "no information"
        ]

        return any(
            phrase in answer
            for phrase in unknown_phrases
        )

    return expected_answer.lower() in answer

def evaluate_prompt(llm_test_queries, model, embeddings, docs, prompt_template):
    correct = 0

    for test in llm_test_queries:
        results = vector_search(
            test["query"],
            model,
            embeddings,
            docs,
            num_results=5
        )

        answer = generate_answer(
            test["query"],
            results,
            prompt_template
        )

        is_correct = answer_is_correct(
            answer,
            test["expected_answer"]
        )

        if is_correct:
            correct += 1

        print(
            test["query"],
            "->",
            "CORRECT" if is_correct else "INCORRECT"
        )
        print("Answer:", answer)
        print()

    return correct / len(llm_test_queries)

def main():
    docs=load_docs(DOCUMENTS_FILE)
    index=build_text_index(docs)
    model=load_embedding_model()
    embeddings=load_embeddings(EMBEDDINGS_FILE)

    hitrate1 = evaluate_text_search(test_queries, index)
    hitrate2 = evaluate_vector_search(test_queries, model, embeddings, docs)
    hitrate3 = evaluate_hybrid_search(test_queries, index, model, embeddings, docs)

    mrr1 = evaluate_text_mrr(test_queries, index)
    mrr2 = evaluate_vector_mrr(test_queries, model, embeddings, docs)
    mrr3 = evaluate_hybrid_mrr(test_queries, index, model, embeddings, docs)


    print(f"\nText Search Hit Rate@5: {hitrate1:.2f}")
    print(f"Text Search MRR@5: {mrr1:.2f}")
    print(f"\nVector Search Hit Rate@5: {hitrate2:.2f}")
    print(f"Vector Search MRR@5: {mrr2:.2f}")
    print(f"\nHybrid Search Hit Rate@5: {hitrate3:.2f}")
    print(f"Hybrid Search MRR@5: {mrr3:.2f}")

    score_a=evaluate_prompt(llm_test_queries,model,embeddings,docs,PROMPT_A)
    score_b=evaluate_prompt(llm_test_queries,model,embeddings,docs,PROMPT_B)

    print(f"Prompt A Accuracy: {score_a:.2f}")
    print(f"Prompt B Accuracy: {score_b:.2f}")

if __name__=="__main__":
    main()