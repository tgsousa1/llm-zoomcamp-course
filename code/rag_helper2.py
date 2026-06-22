INSTRUCTIONS = """
Your task is to answer questions about the content of the files provided
in the documents.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE='''
Question: {question}

Context:
{context}
'''.strip()

class RAGbase:
    def __init__(
            self,
            index,
            llm_client,
            instructions=INSTRUCTIONS,
            prompt_template=USER_PROMPT_TEMPLATE,
            model='gpt-5.4-mini'
            ):

            self.index=index
            self.llm_client=llm_client
            self.instructions=instructions
            self.prompt_template=prompt_template
            self.model=model


    def search(self, query, num_results=5):

        return self.index.search(
            query,
            num_results=num_results,
        )

    def build_context(self, search_results):
        lines=[]

        for doc in search_results:
            lines.append("Filename" + doc["filename"])
            lines.append("Content: "+doc["content"])
            lines.append("")
        
        return "\n".join(lines).strip()


    def build_prompt(self,query,search_results):
        context=self.build_context(search_results)
        return self.prompt_template.format(
             question=query,context=context
        )


    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response

    def rag(self, query):
        search_results=self.search(query)
        prompt=self.build_prompt(query, search_results)
        answer=self.llm(prompt)

        return answer