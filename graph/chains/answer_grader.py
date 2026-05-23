"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/chains/answer_grader.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module implements a structured evaluation chain using Pydantic and LLM
    Structured Output. It assesses whether a generated answer successfully
    addresses and resolves the initial user question, providing a binary validation.

KENDİME NOT:
    Bu zincirin 'hallucination_grader'dan en büyük farkı şudur:
    Halüsinasyon kontrolü 'Cevap dökümanlara dayanıyor mu?' diye bakarken,
    bu kontrol 'Cevap dökümanlara dayansa bile kullanıcının sorusuna çare
    oldu mu?' sorusuna bakar. (Örn: Soru "Hamburger nasıl yapılır?",
    Cevap "Hamburger çok lezzetlidir" ise halüsinasyon yoktur ama soruya cevap da değildir.)
================================================================================
"""

from graph.config import llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. VERİ ŞEMASI TANIMLANMASI (PYDANTIC MODEL)
# ------------------------------------------------------------------------------
class GradeAnswer(BaseModel):
    """
    LLM'den dönmesi zorunlu kılınan katı JSON şeması.
    """
    # KENDİME NOT: Pydantic sayesinde LLM bize uzun uzadıya cümleler kuramaz.
    # Sadece ve sadece True veya False içeren net bir veri modeli döner.
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'",
    )


# ------------------------------------------------------------------------------
# 2. YAPISAL ÇIKTI ENTEGRASYONU (STRUCTURED OUTPUT)
# ------------------------------------------------------------------------------
# KENDİME NOT: En güncel .with_structured_output() metodunu kullanarak
# gpt-4o-mini modelini yukarıdaki Pydantic şemasına kelepçeliyoruz.
structured_llm_grader = llm.with_structured_output(GradeAnswer)


# ------------------------------------------------------------------------------
# 3. PROMPT ŞABLONU (PROMPT TEMPLATE)
# ------------------------------------------------------------------------------
system_prompt = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.
"""

# {question} -> Kullanıcının sorduğu ilk soru
# {generation} -> LLM'in dökümanlara bakarak ürettiği yanıt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)


# ------------------------------------------------------------------------------
# 4. ZİNCİRİN OLUŞTURULMASI (LCEL PIPELINE)
# ------------------------------------------------------------------------------
# KENDİME NOT: LangChain Expression Language (LCEL) | operatörünü kullanıyoruz.
# Girdiler önce prompt şablonuna oturur, oradan çıkan metin yapısal modele gider
# ve bize doğrudan GradeAnswer nesnesi döner. graph.py içinde invoke edilmeye hazırdır.
answer_grader = answer_prompt | structured_llm_grader