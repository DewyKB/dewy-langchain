import random
import string
from time import sleep
import pytest

from dewy_client import Client
from dewy_client.api.kb import add_document, add_collection, get_document_status
from dewy_client.models import AddDocumentRequest, CollectionCreate, IngestState

from dewy_langchain.retriever import DewyRetriever

BASE_URL="http://localhost:8000"

def test_basic_retriever():
    client = Client(base_url=BASE_URL)

    # Create a random collection
    collection = "".join(random.choices(string.ascii_lowercase, k=5))
    add_collection.sync(client=client, body=CollectionCreate(
        name = collection,
    ))

    # Add a document.
    document = add_document.sync(client=client, body=AddDocumentRequest(
        collection=collection,
        url="https://raw.githubusercontent.com/DewyKB/dewy/main/test_data/nearly_empty.pdf"
    ))

    status = get_document_status.sync(document.id, client=client)
    while status.ingest_state == IngestState.PENDING:
        sleep(0.2)
        status = get_document_status.sync(document.id, client=client)

    retriever = DewyRetriever.for_collection(collection, base_url=BASE_URL)
    results = retriever.invoke("nearly empty")

    assert len(results) == 1
    assert results[0].page_content == "This is a nearly empty PDF to test extraction and embedding."

