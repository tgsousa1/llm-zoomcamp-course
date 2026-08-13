from openai import OpenAI
from dotenv import load_dotenv
import os
from monitoring import tracer, llm_input_tokens, llm_output_tokens, llm_total_tokens, rag_duration, rag_errors, rag_requests
import time


from search import (
    load_docs,
    build_text_index,
    load_embedding_model,
    load_embeddings,
    hybrid_search,
    text_search,
    vector_search,
    DOCUMENTS_FILE,
    EMBEDDINGS_FILE
)

load_dotenv()

client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

INSTRUCTIONS = """
You are a movie information assistant.

Answer the user's question using ONLY the provided context.

- Pay attention to the keywords in the user's query.
- For questions about actors, actresses, or directors, look for the relevant information in the movie content.
- If the answer cannot be found in the context, say that you don't know.
- Don't suggest movies with similar titles.
- Keep your answers concise and factual.
"""

PROMPT_TEMPLATE = """
QUESTION:
{question}

CONTEXT:
{context}
"""

def build_context(search_results):
    lines=[]

    for movie in search_results:
        lines.append(movie["content"])
        lines.append("")
        
    return "\n".join(lines).strip()

def build_prompt(query, search_results):
    context = build_context(search_results)

    return PROMPT_TEMPLATE.format(
        question=query,
        context=context
    )

def llm(prompt):
    with tracer.start_as_current_span("llm_call") as span:

        span.set_attribute("model",MODEL)

        client=OpenAI()

        response= client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "developer",
                    "content": INSTRUCTIONS
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )
        
        if response.usage:

            usage = response.usage

            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            total_tokens = usage.total_tokens

            span.set_attribute("input_tokens", input_tokens)
            span.set_attribute("output_tokens", output_tokens)
            span.set_attribute("total_tokens", total_tokens)

            llm_input_tokens.add(input_tokens)
            llm_output_tokens.add(output_tokens)
            llm_total_tokens.add(total_tokens)

        return response.output_text

def rag(query, model, embeddings, documents):

    start_time=time.perf_counter()
    rag_requests.add(1)

    try:
        with tracer.start_as_current_span("rag") as span:

            span.set_attribute("query",query)
            span.set_attribute("num_documents", len(documents))
            span.set_attribute("retrieval_method", "vector")

            search_results=vector_search(
                query=query,
                model=model,
                embeddings=embeddings,
                documents=documents,
                num_results=5
            )

            span.set_attribute("num_results",len(search_results))

            prompt=build_prompt(query, search_results)

            answer=llm(prompt)

            return answer, search_results
    except Exception:
        rag_errors.add(1)
        raise

    finally:
        duration = time.perf_counter() - start_time
        rag_duration.record(duration)

#--------testing---------

# def main():
#     docs = load_docs(DOCUMENTS_FILE)

#     text_index = build_text_index(docs)

#     model = load_embedding_model()

#     embeddings = load_embeddings(EMBEDDINGS_FILE)

#     query = input("Question: ")

#     answer = rag(
#         query=query,
#         index=text_index,
#         model=model,
#         embeddings=embeddings,
#         documents=docs
#     )

#     print("\nAnswer:\n")
#     print(answer)


# if __name__ == "__main__":
#     main()