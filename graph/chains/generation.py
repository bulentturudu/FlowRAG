"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/chains/generation_chain.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    This module creates the final text generation chain for the FlowRAG system.
    It dynamically pulls the industry-standard "rlm/rag-prompt" template from
    LangSmith Hub, combines it with our centralized LLM config, and pipes the
    result into a String Output Parser to return clean textual answers.

KENDİME NOT:
    Burası sistemin 'konuşma' odasıdır. Kendisine beslenen doğrulanmış
    belgeler (context) ile kullanıcının sorusunu (question) birleştirerek
    uydurma yapmadan, tamamen kanıtlara dayalı bir cevap metni sentezler.
================================================================================
"""

from graph.config import llm
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from dotenv import load_dotenv

# .env içerisindeki LangSmith API ve OpenAI anahtarlarını yükle
load_dotenv()

# ------------------------------------------------------------------------------
# 1. LANGSMITH HUB ENTEGRASYONU (DYNAMIC PROMPT INGESTION)
# ------------------------------------------------------------------------------
# KENDİME NOT: Prompt şablonunu kodun içine yazıp kirletmek yerine, LangChain
# topluluğunun kabul ettiği en optimize RAG promptu olan "rlm/rag-prompt" şablonunu
# LangSmith bulutundan dinamik olarak çekiyoruz. Bu sayede prompt güncellense bile
# kodumuzu değiştirmek zorunda kalmıyoruz.
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)


# ------------------------------------------------------------------------------
# 2. ÜRETİM ZİNCİRİNİN OLUŞTURULMASI (LCEL PIPELINE)
# ------------------------------------------------------------------------------
# KENDİME NOT: LangChain Expression Language (LCEL) zinciridir.
# 1. 'prompt' girdileri (context ve question) alır, şablona oturtur.
# 2. 'llm' (gpt-4o-mini) bu şablonu okur ve profesyonel yanıtı üretir.
# 3. 'StrOutputParser()' gelen karmaşık LLM yanıt nesnesinin içinden sadece
#     düz metni (string) cımbızla çeker ve bize tertemiz teslim eder.
generate_chain = prompt | llm | StrOutputParser()