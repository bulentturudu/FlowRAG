"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/config.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    Centralized configuration module for Large Language Models (LLM).
    Initializes and exports the primary model instance used across all graph
    nodes and grading chains, ensuring parameter consistency.

KENDİME NOT:
    Projede kullanılan tüm yapay zeka zekası bu tek merkezden (`llm`) beslenir.
    İleride daha güçlü bir modele geçmek (örn: gpt-4o) veya yaratıcılığı
    artırmak (temperature değiştirmek) istersen sadece bu dosyayı güncellemen yeterlidir.
================================================================================
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env içerisindeki API anahtarlarını yükle
load_dotenv()

# ------------------------------------------------------------------------------
# MERKEZİ LLM YAPILANDIRMASI
# ------------------------------------------------------------------------------
# KENDİME NOT:
# temperature=0: Yanıtların tamamen deterministik (tutarlı, kesin ve uydurmadan uzak)
# olmasını sağlar. RAG sistemlerinde halüsinasyonu engellemek için 0 olması şarttır.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)