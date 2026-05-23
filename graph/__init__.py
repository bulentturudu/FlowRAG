"""
================================================================================
PROJECT: FlowRAG (Advanced Graph-Based RAG System)
FILE: graph/nodes/__init__.py
AUTHOR: Bulent Turudu
DESCRIPTION:
    Package initialization file for graph nodes.
    It acts as a centralized clean interface (Facade) that exposes all workflow
    node functions, simplifying imports inside the main graph configuration.

KENDİME NOT:
    Bu dosya sayesinde dışarıdaki bir modülden (örneğin graph.py) düğümlere
    ulaşmak istediğimizde her dosya için ayrı ayrı import satırı yazmak yerine,
    doğrudan `from graph.nodes import retrieve, generate` şeklinde toplu ve
    tertemiz bir çağrı yapabiliyoruz. Kod okunabilirliğini inanılmaz artırır.
================================================================================
"""

from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search

# ------------------------------------------------------------------------------
# PAKET DIŞA AKTARIM YAPILANDIRMASI (__all__)
# ------------------------------------------------------------------------------
# KENDİME NOT: __all__ listesi, bu klasörden (paket) dışarıya "nelerin sızabileceğini"
# açıkça ilan eder. Başka bir dosya `from graph.nodes import *` yazarsa, sadece ve
# sadece bu listedeki fonksiyonlar içeri alınır; gizli kalması gereken yapılar korunur.
__all__ = [
    "generate",
    "grade_documents",
    "retrieve",
    "web_search"
]