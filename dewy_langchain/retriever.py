from __future__ import annotations

from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import Any, Coroutine, List, Optional

from dewy_client import Client
from dewy_client.api.kb import retrieve_chunks
from dewy_client.models import RetrieveRequest, TextResult

class Retriever(BaseRetriever):
    """Retriever using Dewy for knowledege management.

    Example:
      .. code-block:: python

        from dewy_langchain import Retriever
        retriever = Retriever.for_collection("main")
    """

    client: Client
    collection: str

    def __init__(self,
                 client: Client,
                 collection: str) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def for_collection(
        collection: str = "main",
        *,
        base_url: Optional[str] = None,
    ) -> Retriever:
        pass

    def _make_request(self, query: str) -> RetrieveRequest:
        return RetrieveRequest(
            collection_id=self.collection_id,
            query=query,
            include_image_chunks=False,
        )

    def _make_document(self, chunk: TextResult) -> Document:
        return Document(page_content=chunk.text, metadata = { "chunk_id": chunk.chunk_id })

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        retrieved = retrieve_chunks.sync(client=self.client, body=self._make_request(query))
        return [self._make_document(chunk) for chunk in retrieved.text_results]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> Coroutine[Any, Any, List[Document]]:
        retrieved = await retrieve_chunks.asyncio(client=self.client, body=self._make_request(query))
        return [self._make_document(chunk) for chunk in retrieved.text_results]