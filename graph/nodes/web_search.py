"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/nodes/web_search.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module implements the Web Search Node using Tavily Search API.
    It extracts external real-time data when local vector store information
    is missing or deemed irrelevant, safely normalizes different Tavily API
    response types into standard LangChain Documents, and appends them to the state.

graph.py'deki Akış Yeri:
    route_question (websearch) ──► WEBSEARCH ──► GENERATE
    GRADE_DOCUMENT (web_search=True) ──► WEBSEARCH ──► GENERATE
================================================================================
"""

from typing import Any, Dict
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

# ------------------------------------------------------------------------------
# TAVILY SEARCH MOTORU TANIMLANMASI
# ------------------------------------------------------------------------------
# KENDİME NOT: max_results=3 diyerek LLM'e sadece en alakalı ilk 3 internet
# sonucunu paslıyoruz. Bu hem token maliyetini düşürür hem de gürültüyü azaltır.
# Çalışması için .env dosyasında TAVILY_API_KEY tanımlı olmalıdır.
web_search_tool = TavilySearch(max_results=3)


# ==============================================================================
# WEB SEARCH DÜĞÜMÜ (NODE FUNCTION)
# ==============================================================================
def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Kullanıcının sorusunu internette aratır, gelen ham sonuçları temizleyerek
    LangChain Document objelerine dönüştürür ve mevcut döküman listesine ekler.
    """
    print("---- WEB SEARCH ----")

    question = state["question"]
    # Eğer daha önce yerel veritabanından çekilmiş dökümanlar varsa listeyi bozma,
    # üstüne ekleme yapmak için state'den mevcut listeyi güvenle çek.
    documents = state.get("documents", [])

    # Tavily arama motorunu tetikliyoruz
    search_results = web_search_tool.invoke(
        {
            "query": question
        }
        )

    # İnternetten süzülecek temiz dökümanların birikeceği geçici liste
    web_documents = []

    # ------------------------------------------------------------------------------
    # 🛡️ GÜVENLİK SÜZGECİ (DEFENSIVE PROGRAMMING / TYPE GUARDING)
    # ------------------------------------------------------------------------------
    # KENDİME NOT: API'ler bazen güncellenir veya farklı veri yapıları dönebilir.
    # Kodun çalışma anında (runtime) çökmesini engellemek için 3 olasılığı da yönetiyoruz:

    if isinstance(search_results, str):
        # 1. OLASILIK: Tavily doğrudan bir metin özeti döndüyse
        web_documents.append(
            Document(
                page_content=search_results,
                metadata={
                    "source": "Tavily Search Summary"
                }
            )
        )

    elif isinstance(search_results, list):
        # 2. OLASILIK: Doğrudan sonuçların listesi döndüyse
        for result in search_results:
            content = result.get("content") or result.get("snippet") or str(result)
            source_url = result.get("url", "Internet Source")
            web_documents.append(
                Document(
                    page_content=content, metadata={
                        "source": source_url
                    }
                    )
            )

    elif isinstance(search_results, dict):
        # 3. OLASILIK: Karmaşık bir sözlük ve içinde 'results' anahtarı döndüyse
        results_list = search_results.get("results", [])
        for result in results_list:
            content = result.get("content") or result.get("snippet") or str(result)
            source_url = result.get("url", "Internet Source")
            web_documents.append(
                Document(
                    page_content=content, metadata={
                        "source": source_url
                    }
                    )
            )

    # ACİL DURUM BUTONU: Eğer yukarıdaki formatlara uymayan garip bir nesne döndüyse,
    # veriyi kaybetmemek için nesneyi string'e çevirip içeri alıyoruz.
    if not web_documents and search_results:
        web_documents.append(
            Document(
                page_content=str(search_results),
                metadata={
                    "source": "Tavily Raw"
                }
            )
        )

    # ------------------------------------------------------------------------------
    # BİLGİ TAKVİYESİ VE STATE GÜNCELLEMESİ
    # ------------------------------------------------------------------------------
    # KENDİME NOT: İnternetten topladığımız taze dokümanları, mevcut doküman listesinin
    # arkasına ekliyoruz (.extend ile yerinde modifikasyon). Böylece sistem hem yerel
    # verileri hem de internet verilerini harmanlayarak GENERATE düğümüne gönderecek.
    documents.extend(web_documents)

    return {
        "documents": documents,
        "question": question
    }