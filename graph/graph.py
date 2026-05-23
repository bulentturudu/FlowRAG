"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/graph.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    The core orchestration engine of FlowRAG. This file defines the StateGraph
    workflow using LangGraph. It establishes the structural nodes, connects them
    via static and conditional edges, and embeds runtime evaluation logic
    (Hallucination & Answer Grading) to execute a Corrective RAG (CRAG) pipeline.

Akış Şeması (Workflow Diagram):
    Soru (Query)
      │
      ▼
  route_question ──► WEBSEARCH ──────────────────┐
      │                                           │
      ▼                                           │
   RETRIEVE                                       │
      │                                           │
      ▼                                           │
 GRADE_DOCUMENT ──► WEBSEARCH (Yetersiz Belge)  ┘
      │ (Belgeler Yeterli)                        │
      ▼                                           │
   GENERATE ◄────────────────────────────────────┘
      │
      ├──► "useful"        → END (Başarılı Sonuç)
      ├──► "not supported" → GENERATE (Halüsinasyon - Yeniden Dene)
      └──► "not useful"    → WEBSEARCH (Cevap Yetersiz - İnternete Sor)
================================================================================
"""

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router
from graph.node_constants import GENERATE, GRADE_DOCUMENT, RETRIEVE, WEBSEARCH
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search
from graph.state import GraphState

# API Anahtarlarının düğümler tarafından okunabilmesi için ortamı yükle
load_dotenv()


# ==============================================================================
# KOŞULLU KARAR FONKSİYONLARI (CONDITIONAL EDGES LOGIC)
# ==============================================================================

def decide_to_generate(state: GraphState) -> str:
    """
    Belge Değerlendirme (GRADE_DOCUMENT) sonrasında akışın yönünü belirler.
    KENDİME NOT: Eğer dokümanlar incelendiğinde 'web_search' bayrağı True
    set edildiyse (yani yerel veri alakasızsa) internete çıkar, yoksa üretime geçer.
    """
    print("----ASSESS GRADED DOCUMENTS----")
    if state["web_search"]:
        print("----WEBSEARCH----")
        return WEBSEARCH
    return GENERATE


def grade_generation_grounded_in_documents_and_questions(state: GraphState) -> str:
    """
    Üretilen cevabı (generation) iki aşamalı sıkı bir teste tabi tutar:
    1. Halüsinasyon Kontrolü (Grounded in documents?)
    2. Soru Yanıtlama Kontrolü (Addresses question?)
    """
    print("----CHECK HALLUCINATION----")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    loop_count = state.get("loop_count", 0)  # Akıllı sonsuz döngü sayacını oku

    # --- 1. AŞAMA: HALÜSİNASYON KONTROLÜ ---
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if score.binary_score:
        print("GENERATION IS GROUNDED IN DOCUMENTS")

        # --- 2. AŞAMA: METİN/SORU UYUM KONTROLÜ ---
        score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if score.binary_score:
            print("GENERATION ADDRESSES QUESTION")
            return "useful"  # Her şey mükemmel, akış başarıyla biter (END)
        else:
            print("GENERATION DOES NOT ADDRESS THE QUESTION")
            return "not useful"  # Cevap var ama yetersiz, internete yönlendir (WEBSEARCH)
    else:
        print("GENERATION IS NOT GROUNDED IN DOCUMENTS")

        # 🛠️ GÜVENLİK DUVARI BARİYERİ
        # KENDİME NOT: Eğer model uydurmaya (halüsinasyona) başladıysa ve bunu
        # 3 kereden fazla üst üste denediyse, OpenAI bütçesini korumak için
        # akışı zorla durdurup "I don't know" mesajını final state'e kilitliyoruz.
        if loop_count >= 3:
            print("---- MAX GENERATION ATTEMPTS REACHED. FORCING END TO AVOID INFINITE LOOP ----")
            return "useful"  # Sistemi kibarca END noktasına pasla

        return "not supported"  # 3 denemeden azsa, tekrar üretmesi için GENERATE'e gönder


def route_question(state: GraphState) -> str:
    """
    Giriş Yönlendiricisi (Intent Router). Soruyu analiz ederek grafiğin
    nereden başlayacağını seçer. Genel kültür ise WEBSEARCH, yerel veri ise RETRIEVE.
    """
    print("----ROUTE QUESTION----")
    question = state["question"]
    source = question_router.invoke({"question": question})

    if source.datasource == "websearch":
        print("----WEBSEARCH----")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        return RETRIEVE


# ==============================================================================
# GRAFİK MİMARİSİ VE BAĞLANTILARIN TANIMLANMASI (STATE GRAPH)
# ==============================================================================

# Ortak State şemamızı kullanarak ana grafiği başlatıyoruz
workflow = StateGraph(GraphState)

# 1. DÜĞÜMLERİN (NODES) GRAFİĞE KAYDEDİLMESİ
workflow.add_node(RETRIEVE, retrieve)               # Vektör veritabanı okuyucu
workflow.add_node(GRADE_DOCUMENT, grade_documents)  # LLM Tabanlı filtreleyici
workflow.add_node(GENERATE, generate)               # Yanıt üretim merkezi
workflow.add_node(WEBSEARCH, web_search)            # Güncel internet arama motoru

# 2. BAŞLANGIÇ NOKTASININ KOŞULA BAĞLANMASI (ENTRY POINT)
# KENDİME NOT: Sistem ilk çalıştığında route_question çalışır ve gelen
# sonuca göre grafiğe ya RETRIEVE ya da WEBSEARCH kapısından giriş yaptırır.
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    }
)

# 3. STATİK VE KOŞULLU BAĞLANTILARIN (EDGES) ÖRÜLMESİ
# Yerelden veri çekildiyse, istisnasız her zaman filtreleme düğümüne git
workflow.add_edge(RETRIEVE, GRADE_DOCUMENT)

# Filtreleme bittiğinde döküman kalitesine göre internete mi yoksa LLM'e mi gidileceğini seç
workflow.add_conditional_edges(
    GRADE_DOCUMENT,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    }
)

# Yanıt üretildiğinde kalite kontrol süzgeçlerinin sonucuna göre yön tayin et
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_questions,
    {
        "not supported": GENERATE,   # Halüsinasyon var -> Sil baştan üret
        "useful": END,               # Kusursuz yanıt -> Akışı Sonlandır
        "not useful": WEBSEARCH,     # Eksik bilgi -> İnternetten takviye al
    }
)

# İnternet araması tamamlandığında, toplanan yeni verilerle cevap üretmesi için LLM'e git
workflow.add_edge(WEBSEARCH, GENERATE)

# KENDİME NOT: Eski kodda bulunan statik `workflow.add_edge(GENERATE, END)` satırı,
# yukarıdaki koşullu (conditional) yönlendirmeyle çakışma riski yarattığı için kaldırıldı.
# Artık GENERATE düğümünden sonra tek karar mercii üstteki kalite kontrol fonksiyonudur.

# 4. DERLEME (COMPILATION)
# Tüm düğüm ve kenarları birleştirip çalıştırılabilir LangGraph uygulamasını oluşturur
app = workflow.compile()

# KENDİME NOT: İleride grafiğin görsel şemasını (PNG) çıkartıp incelemek istersen alttaki satırı açabilirsin:
# app.get_graph().draw_mermaid_png(output_file_path="graph_architecture.png")