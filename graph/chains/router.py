"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/chains/router.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module implements an Intent Router using strict Pydantic Literal types.
    It analyzes the user's initial question at the entry point of the graph and
    deterministically routes it either to the local vectorstore (ChromaDB) or
    the external search engine (Tavily), minimizing execution latency.

KENDİME NOT:
    Burası sistemin giriş kapısıdır. Eğer kullanıcı 'hamburger tarifi' veya
    'hava durumu' gibi yerel dökümanlarda olmayan genel bir soru sorarsa,
    router bunu anlar ve doğrudan 'websearch' çıktısı verir. Böylece yerel
    veritabanını (Chroma) boşuna tarayıp vakit kaybetmeyiz.
================================================================================
"""

from graph.config import llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal


# ------------------------------------------------------------------------------
# 1. YÖNLENDİRME ŞEMASI TANIMLANMASI (PYDANTIC MODEL)
# ------------------------------------------------------------------------------
class RouteQuery(BaseModel):
    """
    Kullanıcı sorusunun hangi veri kaynağına aktarılacağını belirleyen katı şema.
    """
    # KENDİME NOT: Literal["vectorstore", "websearch"] kullanarak modelin bu iki
    # kelime dışında kafasına göre üçüncü bir seçenek (örneğin "google", "database")
    # üretmesini kesin olarak engelliyoruz. Tip güvenliği için harika bir yöntemdir.
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to the web search or a vectorstore datasource",
    )


# ------------------------------------------------------------------------------
# 2. YAPISAL ÇIKTI ENTEGRASYONU (STRUCTURED OUTPUT)
# ------------------------------------------------------------------------------
# KENDİME NOT: .with_structured_output() metoduna Pydantic şemasını vererek
# gpt-4o-mini modelini bu iki kelimeden birini dönmeye zorluyoruz.
structured_llm_router = llm.with_structured_output(RouteQuery)


# ------------------------------------------------------------------------------
# 3. YÖNLENDİRİCİ PROMPT ŞABLONU (ROUTER PROMPT)
# ------------------------------------------------------------------------------
# KENDİME NOT: Modele yerel veritabanımızda ne olduğunu açıkça öğretiyoruz.
# İçeride Lilian Weng'in makaleleri (agents, prompt engineering, adversarial attacks)
# olduğu için bu konuları vectorstore'a, diğer her şeyi ise web-search'e paslatıyoruz.
system_prompt = """ 
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search.
"""

# {question} -> Kullanıcının `main.py` üzerinden gönderdiği ham soru
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)


# ------------------------------------------------------------------------------
# 4. ZİNCİRİN OLUŞTURULMASI (LCEL PIPELINE)
# ------------------------------------------------------------------------------
# KENDİME NOT: LangChain Expression Language (LCEL) boru hattı.
# Soru prompt ile birleşir, yapılandırılmış LLM'e gider ve bize 'datasource' alanını döndürür.
# graph.py içerisindeki `route_question` fonksiyonunda ilk bu tetiklenir.
question_router = route_prompt | structured_llm_router