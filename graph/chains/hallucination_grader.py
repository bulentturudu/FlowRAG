"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/chains/hallucination_grader.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module implements a structured hallucination checker using Pydantic.
    It takes the retrieved documents (facts) and the LLM-generated answer,
    then performs a strict binary assessment to ensure the answer is strictly
    grounded in the provided evidence, eliminating model hallucinations.

KENDİME NOT:
    Bu zincir, modelin uydurma yapmasını engelleyen ana filtredir.
    Eğer model dökümanda geçmeyen bir bilgiyi sırf kulağa güzel geliyor diye
    cevaba eklerse, bu zincir durumu fark eder ve 'False' döner. `graph.py`
    ise bu sinyali alıp modeli yeniden üretim yapmaya (GENERATE) zorlar.
================================================================================
"""

from graph.config import llm
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. VERİ ŞEMASI TANIMLANMASI (PYDANTIC MODEL)
# ------------------------------------------------------------------------------
class GradeHallucinations(BaseModel):
    """
    LLM'den dönmesi zorunlu kılınan katı JSON şeması.
    """
    # KENDİME NOT: Pydantic şeması sayesinde modelden sadece True (uydurma yok,
    # dökümanlara bağlı) veya False (uydurma var / halüsinasyon) yanıtı alırız.
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'",
    )


# ------------------------------------------------------------------------------
# 2. YAPISAL ÇIKTI ENTEGRASYONU (STRUCTURED OUTPUT)
# ------------------------------------------------------------------------------
# KENDİME NOT: .with_structured_output() kullanarak gpt-4o-mini modelini
# yukarıdaki GradeHallucinations şemasına bağlıyoruz. Çıktı doğrudan JSON formatında süzülür.
structured_llm_grader = llm.with_structured_output(GradeHallucinations)


# ------------------------------------------------------------------------------
# 3. PROMPT ŞABLONU (PROMPT TEMPLATE)
# ------------------------------------------------------------------------------
system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# {documents}  -> Kaynak dökümanlar (Gerçekler kümesi)
# {generation} -> LLM'in ürettiği taslak yanıt
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)


# ------------------------------------------------------------------------------
# 4. ZİNCİRİN OLUŞTURULMASI (LCEL PIPELINE)
# ------------------------------------------------------------------------------
# KENDİME NOT: LangChain Expression Language (LCEL) | operatörü ile bağlanmıştır.
# Girdiler önce şablona oturur, ardından yapısal modele iletilir ve bize
# doğrudan GradeHallucinations nesnesi döner. graph.py içinde koşmaya hazırdır.
hallucination_grader = hallucination_prompt | structured_llm_grader