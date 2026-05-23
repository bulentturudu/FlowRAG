"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/nodes/retrieve.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module implements the primary Document Retrieval Node for FlowRAG.
    It instantiates a persistent Chroma vectorstore receiver and queries
    the local database using OpenAI Embeddings to extract the top semantically
    similar document chunks matching the user's inquiry.

graph.py'deki Akış Yeri:
    route_question (vectorstore) ──► RETRIEVE ──► GRADE_DOCUMENT
================================================================================
"""

from typing import Any, Dict
from langchain_chroma import Chroma  # 🛠️ GÜNCEL paket yapısı kullanılıyor
from langchain_openai import OpenAIEmbeddings
from graph.state import GraphState

# ------------------------------------------------------------------------------
# VEKTÖR VERİTABANI BAĞLANTISI VE RETRIEVER TANIMI
# ------------------------------------------------------------------------------
# KENDİME NOT: `ingestion.py` dosyasında 'rag-chroma' koleksiyon adı ve OpenAIEmbeddings
# kullanarak diskteki `./.chroma` klasörüne yazdığımız index yapısını burada ayağa kaldırıyoruz.
vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./.chroma"
)

# .as_retriever() metodu ile vektör deposunu otomatik bir arama motoruna dönüştürüyoruz.
retriever = vectorstore.as_retriever()


# ==============================================================================
# RETRIEVE DÜĞÜMÜ (NODE FUNCTION)
# ==============================================================================
def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Kullanıcının sorusunu alır, yerel ChromaDB indekslerinde semantik arama yapar
    ve bulduğu en alakalı döküman listesini grafiğin ortak hafızasına kilitler.
    """
    print("----RETRIEVE----")

    # Grafiğin mevcut durumundan kullanıcının sorduğu soruyu alıyoruz
    question = state["question"]

    # ------------------------------------------------------------------------------
    # SEMANTİK BENZERLİK ARAMASI (SEMANTIC SEARCH)
    # ------------------------------------------------------------------------------
    # KENDİME NOT: retriever.invoke() işlemi arkada sorunun vektörünü çıkartır,
    # veritabanındaki döküman vektörleriyle kosinüs benzerliği (cosine similarity)
    # hesaplaması yapar ve en yakın LangChain Document nesnelerini liste halinde döner.
    documents = retriever.invoke(question)

    # ------------------------------------------------------------------------------
    # STATE GÜNCELLEMESİ VE GERİ DÖNÜŞ
    # ------------------------------------------------------------------------------
    # Bulunan ham döküman listesini bir sonraki düğüm olan `grade_documents` incelesin
    # diye grafiğin ortak hafızasına (State) teslim ediyoruz.
    return {
        "question": question,
        "documents": documents
    }