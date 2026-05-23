"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/nodes/generate.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module defines the execution node for answer generation. It extracts
    the validated documents (context) and the user question from the current
    graph state, invokes the centralized generation chain, and increments
    the loop counter to prevent infinite evaluation cycles.

KENDİME NOT:
    Burası grafiğin metin üreten fabrikasıdır (Node). `generate_chain` isimli
    zinciri tetikler. Her tetiklendiğinde döngü sayacını (`loop_count`)
    otomatik olarak 1 artırır, böylece `graph.py` içindeki halüsinasyon koruma
    bariyeri modelin kaçıncı denemede olduğunu tam olarak takip edebilir.
================================================================================
"""

from typing import Any, Dict
from graph.chains.generation import generate_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Mevcut state içerisindeki dökümanları kullanarak kullanıcının sorusuna
    LLM ile zengin ve tutarlı bir yanıt üretir.
    """
    print("----GENERATE----")
    question = state["question"]
    documents = state["documents"]

    # ------------------------------------------------------------------------------
    # 🛠️ AKILLI DÖNGÜ SAYAÇ YÖNETİMİ
    # ------------------------------------------------------------------------------
    # KENDİME NOT: Eğer sistem grafiğe yeni girdiyse ve 'loop_count' henüz state
    # içinde yoksa varsayılan olarak 0 alırız ve bu ilk üretim denemesi olduğu için
    # üzerine 1 ekleriz. Eğer model halüsinasyon sebebiyle buraya tekrar düştüyse,
    # mevcut sayıyı (örneğin 1'i) alıp 2 yaparız.
    loop_count = state.get("loop_count", 0) + 1

    # ------------------------------------------------------------------------------
    # LLM ZİNCİRİNİN TETİKLENMESİ
    # ------------------------------------------------------------------------------
    # LangSmith Hub'dan çektiğimiz prompt şablonuna 'context' ve 'question' girdilerini
    # besleyerek gpt-4o-mini modelinden nihai metin cevabını istiyoruz.
    generation = generate_chain.invoke(
        {
            "context": documents,
            "question": question
        }
    )

    # ------------------------------------------------------------------------------
    # STATE GÜNCELLEMESİ VE GERİ DÖNÜŞ
    # ------------------------------------------------------------------------------
    # KENDİME NOT: LangGraph mimarisinde bir node'un döndürdüğü bu sözlük (dict),
    # global state içerisindeki ilgili anahtarları otomatik olarak günceller/üzerine yazar.
    return {
        "question": question,
        "documents": documents,
        "generation": generation,
        "loop_count": loop_count  # Güncellenen yeni deneme sayısını state'e mühürlüyoruz
    }