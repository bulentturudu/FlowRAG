"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/state.py
AUTHOR: Bulent Turudu
DESCRIPTION: 
    Defines the global state schema for the LangGraph workflow using TypedDict.
    This class acts as the shared memory (state) that flows through each node 
    and conditional edge in the graph, ensuring structural type safety.

KENDİME NOT: 
    LangGraph mimarisindeki her bir 'Node' (Düğüm), girdi olarak bu GraphState 
    sözlüğünü alır ve çıktı olarak yine bu sözlüğün alanlarını güncelleyerek 
    geri döndürür. Buraya yeni bir anahtar eklemek, tüm grafiğin o veriyi 
    tanımasını sağlar.
================================================================================
"""

from typing import List, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    FlowRAG iş akışı boyunca düğümler arasında aktarılan veri yapısı.
    """

    # Kullanıcının sisteme sorduğu ham soru metni
    question: str

    # LLM (Yapay Zeka) tarafından üretilen nihai yanıt metni
    generation: str

    # Akışta internet aramasının (Tavily) tetiklenip tetiklenmediğini tutan bayrak
    web_search: bool

    # KENDİME NOT: Tip güvenliğini tam sağlamak için ham listeyi List[Document] yaptık.
    # Hem yerel ChromaDB'den gelen hem de internetten (Tavily) süzülen tüm temiz 
    # içerikler LangChain Document nesneleri olarak bu listede birikir.
    documents: List[Document]

    # 🛠️ YENİ / KRİTİK SAYAÇ: Sonsuz döngü bariyeri.
    # KENDİME NOT: Modelin halüsinasyon döngüsüne girip OpenAI cüzdanını eritmesini 
    # engellemek için her üretim (generate) denemesinde bu sayacı 1 artırıyoruz. 
    # Belirlediğimiz limite (örn: 3) ulaştığında akışı zorla sonlandırıyoruz.
    loop_count: int