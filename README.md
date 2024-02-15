# Dewy LangChain Plugin

This package provides Dewy integration for LangChain.

Example:

```
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dewy_langchain import DewyRetriever

retriever = DewyRetriever.for_collection("main", base_url="localhost:8000")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You're a helpful AI assistant. Given a user question and some retrieved content, answer the user question.
            If none of the articles answer the question, just say you don't know.

            Here is the retrieved content:
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

def format_chunks(chunks):
    return "\n\n".join([d.page_content for d in chunks])

chain = (
    { "context": retriever | format_chunks, "question": RunnablePassthrough() }
    | prompt
    | llm
    | StrOutputParser()
)
```