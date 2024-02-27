from __future__ import annotations

from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import Any, Coroutine, List, Optional

from dewy_client import Client
from dewy_client.api.kb import retrieve_chunks
from dewy_client.models import RetrieveRequest, TextResult


class DewyRetriever(BaseRetriever):
    """Retriever using Dewy for knowledege management.

    Example:
      .. code-block:: python

        from dewy_langchain import DewyRetriever
        retriever = DewyRetriever.for_collection("main")
    """

    client: Client
    collection: str

    @staticmethod
    def for_collection(
        collection: str,
        *,
        base_url: Optional[str] = None,
    ) -> DewyRetriever:
        base_url = base_url or "localhost:8000"
        client = Client(base_url)
        return DewyRetriever(client=client, collection=collection)

    def _make_request(self, query: str) -> RetrieveRequest:
        return RetrieveRequest(
            collection=self.collection,
            query=query,
            include_image_chunks=False,
        )

    def _make_document(self, chunk: TextResult) -> Document:
        metadata = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "similarity_score": chunk.score,
        }
        return Document(page_content=chunk.text, metadata=metadata)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        retrieved = retrieve_chunks.sync(
            client=self.client, body=self._make_request(query)
        )
        return [self._make_document(chunk) for chunk in retrieved.text_results]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> Coroutine[Any, Any, List[Document]]:
        retrieved = await retrieve_chunks.asyncio(
            client=self.client, body=self._make_request(query)
        )
        return [self._make_document(chunk) for chunk in retrieved.text_results]
