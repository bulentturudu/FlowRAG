"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/chains/retrieval_grader.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module implements the Document Relevance Grader chain. It utilizes
    OpenAI's Structured Output to evaluate whether a specific retrieved
    document chunk is semantically relevant to the user's initial question,
    returning a strict boolean score.

KENDİME NOT:
    Burası sistemin 'Giriş Kalite Kontrol' departmanıdır. Vektör deposundan
    gelen her dökümanı inceler. Eğer dökümanların tamamı alakasız (False)
    çıkarsa, sistem otomatik olarak 'web_search' bayrağını True yapar ve
    akışı internet aramasına yönlendirir.
================================================================================
"""

from graph.config import llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. VERİ ŞEMASI TANIMLANMASI (PYDANTIC MODEL)
# ------------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    """
    LLM'den dönmesi zorunlu kılınan katı JSON şeması.
    """
    # KENDİME NOT: Tip güvenliğini tam sağlamak ve string eşleşme hatalarından
    # kaçınmak için 'yes/no' yerine doğrudan True/False dönen 'bool' tipini seçtik.
    binary_score: bool = Field(
        description="Is the document relevant to the question? True if yes, False if no."
    )


# ------------------------------------------------------------------------------
# 2. YAPISAL ÇIKTI ENTEGRASYONU (STRUCTURED OUTPUT)
# ------------------------------------------------------------------------------
# KENDİME NOT: .with_structured_output() metodu ile gpt-4o-mini modelini
# yukarıdaki GradeDocuments Pydantic modeline kilitliyoruz.
structured_llm_grader = llm.with_structured_output(GradeDocuments)


# ------------------------------------------------------------------------------
# 3. PROMPT ŞABLONU (PROMPT TEMPLATE)
# ------------------------------------------------------------------------------
# 🛠️ DÜZELTME NOTU: Eski kodda yanlışlıkla hallucination_grader promptu kalmıştı.
# Burayı tamamen dökümanın soruyla olan semantik ve anahtar kelime ilişkisine odakladık.
system_prompt = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the user question, grade it as relevant.
Give a binary score True or False. True means that the document is relevant to the question."""

# {document} -> Vektör deposundan çekilen o anki parça (chunk)
# {question} -> Kullanıcının sorduğu ham soru
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)


# ------------------------------------------------------------------------------
# 4. ZİNCİRİN OLUŞTURULMASI (LCEL PIPELINE)
# ------------------------------------------------------------------------------
# KENDİME NOT: LangChain Expression Language (LCEL) | operatörü ile bağlanmıştır.
# Girdiler şablona oturur, yapısal modele gider ve bize doğrudan GradeDocuments nesnesi döner.
# graph/nodes/grade_documents.py dosyası içinde dökümanları döngüyle eritmek için kullanılır.
retrieval_grader = grade_prompt | structured_llm_grader